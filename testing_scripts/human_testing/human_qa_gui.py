import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import json
import os
import re
import time
from datetime import datetime


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_question(raw: str):
    """Split a MC question string into stem + list of (num, text) options."""
    raw = raw or ""
    opt_re = re.compile(r'(\d)\s*[\)\.:](.+?)(?=\s*\d\s*[\)\.:] |$)', re.DOTALL)
    opts = [(m.group(1).strip(), m.group(2).strip()) for m in opt_re.finditer(raw)]
    first_opt = opt_re.search(raw)
    stem = raw[:first_opt.start()].strip() if first_opt else raw.strip()
    return stem, opts


def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ── custom widgets ────────────────────────────────────────────────────────────

class ModernSlider(tk.Canvas):
    def __init__(self, parent, variable, from_=1, to=10, command=None, **kwargs):
        super().__init__(parent, height=24, bg="#FFFFFF", highlightthickness=0, bd=0, **kwargs)
        self.variable = variable
        self.from_ = from_
        self.to = to
        self.command = command
        
        self.track_color = "#E5E7EB"
        self.fill_color = "#2563EB"
        self.thumb_color = "#FFFFFF"
        self.thumb_outline = "#2563EB"
        
        self.bind("<Configure>", self.draw)
        self.bind("<Button-1>", self.click)
        self.bind("<B1-Motion>", self.drag)
        self.bind("<ButtonRelease-1>", self.release)

    def set_val_from_x(self, x):
        w = self.winfo_width()
        padding = 12
        if w <= 2 * padding: return
        
        if x < padding: x = padding
        if x > w - padding: x = w - padding
        
        ratio = (x - padding) / (w - 2 * padding)
        val = self.from_ + ratio * (self.to - self.from_)
        val = int(round(val))
        
        if self.variable.get() != val:
            self.variable.set(val)
            self.draw()
            if self.command:
                self.command(val)

    def draw(self, event=None):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        padding = 12
        
        # Track
        self.create_line(padding, h//2, w - padding, h//2, fill=self.track_color, width=6, capstyle=tk.ROUND)
        
        val = self.variable.get()
        ratio = (val - self.from_) / (self.to - self.from_)
        x = padding + ratio * (w - 2 * padding)
        
        # Fill
        if x > padding:
            self.create_line(padding, h//2, x, h//2, fill=self.fill_color, width=6, capstyle=tk.ROUND)
            
        # Thumb
        r = 8
        self.create_oval(x - r, h//2 - r, x + r, h//2 + r, fill=self.thumb_color, outline=self.thumb_outline, width=2)
        
    def click(self, event):
        self.set_val_from_x(event.x)
        
    def drag(self, event):
        self.set_val_from_x(event.x)

    def release(self, event):
        self.set_val_from_x(event.x)


# ── main app ──────────────────────────────────────────────────────────────────

class QAReviewer(tk.Tk):
    PAD   = 20
    BG    = "#F8F7F4"
    CARD  = "#FFFFFF"
    ACC   = "#2563EB"       # accent blue
    ACC_L = "#EFF6FF"       # light accent
    TXT   = "#1A1A1A"
    MUTED = "#6B7280"
    BORDER= "#E5E3DC"
    SEL_BG= "#EFF6FF"
    SEL_BD= "#2563EB"
    FONT_BODY   = ("Georgia", 14)
    FONT_SMALL  = ("Helvetica Neue", 11)
    FONT_MONO   = ("Courier New", 12)
    FONT_HEAD   = ("Helvetica Neue", 13, "bold")
    FONT_BTN    = ("Helvetica Neue", 12)

    def __init__(self):
        super().__init__()
        self.title("Brain MRI QA Reviewer")
        self.configure(bg=self.BG)
        self.minsize(760, 560)
        self.resizable(True, True)

        # state
        self.rows      = []
        self.current   = 0
        self.answers   = {}          # idx -> full option string e.g. "1) Answer text"
        self.can_answer = {}         # idx -> "Yes"/"No"
        self.clinically_relevant = {}# idx -> "Yes"/"No"
        self.confidence  = {}        # idx -> float (0.0 to 1.0)
        self.necessary_sequences = {}# idx -> string
        self.comments  = {}          # idx -> free-text comment string
        self.start_time= None
        self.elapsed   = 0.0
        self._timer_id = None
        self.csv_path  = None

        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._build_ui()

        # auto-load if human_dataset.csv sits next to this script
        default = os.path.join(os.path.dirname(os.path.abspath(__file__)), "human_dataset.csv")
        if os.path.exists(default):
            self._load_csv(default)
        else:
            self._show_load_screen()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # top bar
        bar = tk.Frame(self, bg=self.CARD, pady=12, padx=self.PAD)
        bar.pack(fill="x", side="top")
        bar.columnconfigure(1, weight=1)

        tk.Label(bar, text="QA Reviewer", bg=self.CARD, fg=self.TXT,
                 font=("Helvetica Neue", 15, "bold")).grid(row=0, column=0, sticky="w")

        self.timer_lbl = tk.Label(bar, text="", bg=self.CARD, fg=self.MUTED,
                                  font=self.FONT_SMALL)
        self.timer_lbl.grid(row=0, column=2, sticky="e")

        tk.Frame(self, bg=self.BORDER, height=1).pack(fill="x")

        # switcher frame
        self.main = tk.Frame(self, bg=self.BG)
        self.main.pack(fill="both", expand=True)

        self._build_load_screen()
        self._build_landing_screen()
        self._build_quiz_screen()

    def _build_load_screen(self):
        self.load_frame = tk.Frame(self.main, bg=self.BG)
        inner = tk.Frame(self.load_frame, bg=self.CARD, padx=40, pady=40,
                         relief="flat", bd=0,
                         highlightthickness=1, highlightbackground=self.BORDER)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(inner, text="Open human_dataset.csv", bg=self.CARD, fg=self.TXT,
                 font=("Helvetica Neue", 16, "bold")).pack(pady=(0, 6))
        tk.Label(inner, text="Select the CSV file exported by the data pipeline.",
                 bg=self.CARD, fg=self.MUTED, font=self.FONT_SMALL).pack(pady=(0, 24))

        btn_frame = tk.Frame(inner, bg=self.CARD)
        btn_frame.pack()

        self.load_btn = tk.Button(btn_frame, text="Start New Exam...", command=self._open_csv,
                                  bg=self.ACC, fg="white", relief="flat",
                                  font=self.FONT_BTN, padx=24, pady=8,
                                  activebackground="#1D4ED8", activeforeground="white",
                                  cursor="hand2")
        self.load_btn.pack(side="left", padx=5)

        self.resume_btn = tk.Button(btn_frame, text="Resume Save...", command=self._open_save,
                                  bg=self.CARD, fg=self.TXT, relief="flat",
                                  font=self.FONT_BTN, padx=24, pady=8,
                                  highlightthickness=1, highlightbackground=self.BORDER,
                                  activebackground=self.BG, cursor="hand2")
        self.resume_btn.pack(side="left", padx=5)

        self.load_err = tk.Label(inner, text="", bg=self.CARD, fg="#DC2626",
                                 font=self.FONT_SMALL)
        self.load_err.pack(pady=(12, 0))

    def _build_landing_screen(self):
        self.landing_frame = tk.Frame(self.main, bg=self.BG)
        inner = tk.Frame(self.landing_frame, bg=self.CARD, padx=48, pady=48,
                         highlightthickness=1, highlightbackground=self.BORDER)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(inner, text="Brain MRI QA Reviewer", bg=self.CARD, fg=self.TXT,
                 font=("Helvetica Neue", 20, "bold")).pack(pady=(0, 6))

        self.landing_subtitle = tk.Label(inner, text="", bg=self.CARD, fg=self.MUTED,
                                         font=self.FONT_SMALL)
        self.landing_subtitle.pack(pady=(0, 32))

        # stat row
        stats = tk.Frame(inner, bg=self.CARD)
        stats.pack(pady=(0, 32))

        for col, (label, attr) in enumerate([
            ("Questions", "_landing_n_questions"),
            ("Accessions", "_landing_n_accessions"),
            ("File", "_landing_filename"),
        ]):
            cell = tk.Frame(stats, bg=self.ACC_L,
                            highlightthickness=1, highlightbackground="#BFDBFE",
                            padx=20, pady=14)
            cell.grid(row=0, column=col, padx=6)
            val_lbl = tk.Label(cell, text="—", bg=self.ACC_L, fg=self.ACC,
                               font=("Helvetica Neue", 18, "bold"))
            val_lbl.pack()
            tk.Label(cell, text=label, bg=self.ACC_L, fg=self.MUTED,
                     font=self.FONT_SMALL).pack()
            setattr(self, attr + "_lbl", val_lbl)

        tk.Label(inner,
                 text="The timer will start as soon as you press Start exam.\n"
                      "Your answers and total time will be saved automatically when you finish.",
                 bg=self.CARD, fg=self.MUTED, font=self.FONT_SMALL,
                 justify="center").pack(pady=(0, 28))

        btn_frame2 = tk.Frame(inner, bg=self.CARD)
        btn_frame2.pack()

        tk.Button(btn_frame2, text="Start exam", command=self._start_exam,
                  bg=self.ACC, fg="white", relief="flat",
                  font=("Helvetica Neue", 13, "bold"), padx=32, pady=10,
                  activebackground="#1D4ED8", activeforeground="white",
                  cursor="hand2").pack(side="left", padx=5)

        tk.Button(btn_frame2, text="Resume Save...", command=self._open_save,
                  bg=self.CARD, fg=self.TXT, relief="flat",
                  font=("Helvetica Neue", 13, "bold"), padx=32, pady=10,
                  highlightthickness=1, highlightbackground=self.BORDER,
                  activebackground=self.BG, cursor="hand2").pack(side="left", padx=5)

    def _build_quiz_screen(self):
        self.quiz_frame = tk.Frame(self.main, bg=self.BG)

        # ── header row (pinned top) ───────────────────────────────────────────
        hdr = tk.Frame(self.quiz_frame, bg=self.BG, padx=self.PAD, pady=14)
        hdr.pack(fill="x", side="top")

        # accession number
        acc_box = tk.Frame(hdr, bg=self.BG)
        acc_box.pack(side="left")
        tk.Label(acc_box, text="Accession number", bg=self.BG, fg=self.MUTED,
                 font=self.FONT_SMALL).pack(anchor="w")
        self.accession_lbl = tk.Label(acc_box, text="", bg=self.ACC_L, fg=self.ACC,
                                      font=self.FONT_MONO, padx=10, pady=3,
                                      relief="flat", bd=0,
                                      highlightthickness=1, highlightbackground="#BFDBFE")
        self.accession_lbl.pack(anchor="w", pady=(2, 0))

        # progress
        prog_box = tk.Frame(hdr, bg=self.BG)
        prog_box.pack(side="right")
        self.progress_lbl = tk.Label(prog_box, text="", bg=self.BG, fg=self.MUTED,
                                     font=self.FONT_SMALL)
        self.progress_lbl.pack(anchor="e")
        self.prog_bar = ttk.Progressbar(prog_box, length=160, mode="determinate")
        self.prog_bar.pack(anchor="e", pady=(4, 0))

        # ── footer (pinned bottom) ────────────────────────────────────────────
        footer = tk.Frame(self.quiz_frame, bg=self.BG, padx=self.PAD, pady=16)
        footer.pack(fill="x", side="bottom")

        self.prev_btn = tk.Button(footer, text="← Previous", command=self._prev,
                                  bg=self.CARD, fg=self.TXT, relief="flat",
                                  font=self.FONT_BTN, padx=18, pady=7,
                                  highlightthickness=1, highlightbackground=self.BORDER,
                                  activebackground=self.BG, cursor="hand2")
        self.prev_btn.pack(side="left")

        self.hint_lbl = tk.Label(footer, text="", bg=self.BG, fg=self.MUTED,
                                 font=self.FONT_SMALL)
        self.hint_lbl.pack(side="left", padx=16)

        self.next_btn = tk.Button(footer, text="Next →", command=self._next,
                                  bg=self.ACC, fg="white", relief="flat",
                                  font=self.FONT_BTN, padx=18, pady=7,
                                  activebackground="#1D4ED8", activeforeground="white",
                                  cursor="hand2")
        self.next_btn.pack(side="right")

        self.submit_btn = tk.Button(footer, text="Submit now", command=self._submit_now,
                                    bg=self.CARD, fg="#DC2626", relief="flat",
                                    font=self.FONT_BTN, padx=18, pady=7,
                                    highlightthickness=1, highlightbackground="#FECACA",
                                    activebackground="#FEF2F2", cursor="hand2")
        self.submit_btn.pack(side="right")

        self.pause_btn = tk.Button(footer, text="Pause & Save", command=self._pause_now,
                                    bg=self.CARD, fg=self.TXT, relief="flat",
                                    font=self.FONT_BTN, padx=18, pady=7,
                                    highlightthickness=1, highlightbackground=self.BORDER,
                                    activebackground=self.BG, cursor="hand2")
        self.pause_btn.pack(side="right", padx=(0, 8))

        # ── scrollable area (middle) ──────────────────────────────────────────
        self.scroll_canvas = tk.Canvas(self.quiz_frame, bg=self.BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.quiz_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = tk.Frame(self.scroll_canvas, bg=self.BG)
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        def _configure_frame(event):
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        self.scrollable_frame.bind("<Configure>", _configure_frame)

        def _configure_canvas(event):
            self.scroll_canvas.itemconfig(self.canvas_window, width=event.width)
        self.scroll_canvas.bind("<Configure>", _configure_canvas)

        def _on_mousewheel(event):
            if self.scrollable_frame.winfo_reqheight() > self.scroll_canvas.winfo_height():
                self.scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.scroll_canvas.bind('<Enter>', lambda e: self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel))
        self.scroll_canvas.bind('<Leave>', lambda e: self.scroll_canvas.unbind_all("<MouseWheel>"))

        # ── question card ─────────────────────────────────────────────────────
        qcard = tk.Frame(self.scrollable_frame, bg=self.CARD,
                         highlightthickness=1, highlightbackground=self.BORDER,
                         padx=24, pady=20)
        qcard.pack(fill="x", padx=self.PAD, pady=(0, 14))

        tk.Label(qcard, text="Question", bg=self.CARD, fg=self.MUTED,
                 font=self.FONT_HEAD).pack(anchor="w", pady=(0, 8))

        self.q_text = tk.Text(qcard, bg=self.CARD, fg=self.TXT,
                              font=self.FONT_BODY, wrap="word",
                              relief="flat", bd=0, height=4,
                              state="disabled", cursor="arrow")
        self.q_text.pack(fill="x")

        # ── options ───────────────────────────────────────────────────────────
        opts_outer = tk.Frame(self.scrollable_frame, bg=self.BG,
                              padx=self.PAD)
        opts_outer.pack(fill="x")

        self.selected_var = tk.StringVar(value="")
        self.option_frames = []   # (frame, radio) per option
        for _ in range(4):        # up to 4 options
            f = tk.Frame(opts_outer, bg=self.CARD,
                         highlightthickness=1, highlightbackground=self.BORDER,
                         padx=16, pady=12)
            f.pack(fill="x", pady=(0, 8))
            rb = tk.Radiobutton(f, variable=self.selected_var, value="",
                                bg=self.CARD, activebackground=self.SEL_BG,
                                fg=self.TXT, font=self.FONT_BODY,
                                anchor="w", wraplength=580, justify="left",
                                relief="flat", bd=0, cursor="hand2",
                                command=self._on_select)
            rb.pack(fill="x")
            self.option_frames.append((f, rb))

        # ── confidence slider ────────────────────────────────────────────────
        conf_outer = tk.Frame(self.scrollable_frame, bg=self.BG, padx=self.PAD)
        conf_outer.pack(fill="x")

        self.confidence_var = tk.DoubleVar(value=5.0)
        cf = tk.Frame(conf_outer, bg=self.CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=16, pady=8)
        cf.pack(fill="x", pady=(0, 8))
        tk.Label(cf, text="Confidence in your answer:", bg=self.CARD, fg=self.TXT, font=self.FONT_SMALL).pack(side="left")
        
        self.conf_val_lbl = tk.Label(cf, text="50%", bg=self.ACC_L, fg=self.ACC, font=("Helvetica Neue", 12, "bold"), padx=8)
        self.conf_val_lbl.pack(side="right", padx=(10, 0))
        
        def _on_scale_slide(v):
            self.conf_val_lbl.config(text=f"{int(v)*10}%")
            self._on_conf_select()

        self.conf_slider = ModernSlider(cf, variable=self.confidence_var, from_=0, to=10, command=_on_scale_slide)
        self.conf_slider.pack(side="right", fill="x", expand=True, padx=(20, 0))

        # ── additional questions ─────────────────────────────────────────────
        add_outer = tk.Frame(self.scrollable_frame, bg=self.BG, padx=self.PAD)
        add_outer.pack(fill="x")

        self.can_answer_var = tk.StringVar(value="")
        f1 = tk.Frame(add_outer, bg=self.CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=16, pady=8)
        f1.pack(fill="x", pady=(0, 8))
        tk.Label(f1, text="Could the question be answered using the image?", bg=self.CARD, fg=self.TXT, font=self.FONT_SMALL).pack(side="left")
        tk.Radiobutton(f1, text="Yes", variable=self.can_answer_var, value="Yes", bg=self.CARD, fg=self.TXT, cursor="hand2", command=self._on_extra_select).pack(side="right", padx=5)
        tk.Radiobutton(f1, text="No", variable=self.can_answer_var, value="No", bg=self.CARD, fg=self.TXT, cursor="hand2", command=self._on_extra_select).pack(side="right", padx=5)

        self.clinically_relevant_var = tk.StringVar(value="")
        f2 = tk.Frame(add_outer, bg=self.CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=16, pady=8)
        f2.pack(fill="x", pady=(0, 8))
        tk.Label(f2, text="Was the question clinically relevant?", bg=self.CARD, fg=self.TXT, font=self.FONT_SMALL).pack(side="left")
        tk.Radiobutton(f2, text="Yes", variable=self.clinically_relevant_var, value="Yes", bg=self.CARD, fg=self.TXT, cursor="hand2", command=self._on_extra_select).pack(side="right", padx=5)
        tk.Radiobutton(f2, text="No", variable=self.clinically_relevant_var, value="No", bg=self.CARD, fg=self.TXT, cursor="hand2", command=self._on_extra_select).pack(side="right", padx=5)

        self.sequences_var = tk.StringVar(value="")
        f3 = tk.Frame(add_outer, bg=self.CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=16, pady=8)
        f3.pack(fill="x", pady=(0, 8))
        tk.Label(f3, text="Which imaging sequences were necessary to answer the question?", bg=self.CARD, fg=self.TXT, font=self.FONT_SMALL).pack(side="left")
        self.seq_entry = tk.Entry(f3, textvariable=self.sequences_var, bg=self.BG, fg=self.TXT, font=self.FONT_SMALL, relief="flat", highlightthickness=1, highlightbackground=self.BORDER, width=25)
        self.seq_entry.pack(side="right", padx=5)
        self.seq_entry.bind("<FocusOut>", self._save_text_fields)
        self.seq_entry.bind("<KeyRelease>", self._save_text_fields)

        # ── comments ─────────────────────────────────────────────────────────
        ccard = tk.Frame(self.scrollable_frame, bg=self.CARD,
                         highlightthickness=1, highlightbackground=self.BORDER,
                         padx=24, pady=16)
        ccard.pack(fill="x", padx=self.PAD, pady=(0, 8))

        tk.Label(ccard, text="Additional comments", bg=self.CARD, fg=self.MUTED,
                 font=self.FONT_HEAD).pack(anchor="w", pady=(0, 8))

        self.comment_box = tk.Text(ccard, bg=self.BG, fg=self.TXT,
                                   font=self.FONT_SMALL, wrap="word",
                                   relief="flat", bd=0, height=3,
                                   highlightthickness=1,
                                   highlightbackground=self.BORDER,
                                   insertbackground=self.TXT,
                                   padx=8, pady=6)
        self.comment_box.pack(fill="x")
        self.comment_box.bind("<FocusOut>", self._save_text_fields)
        self.comment_box.bind("<KeyRelease>", self._save_text_fields)

    # ── screen switching ──────────────────────────────────────────────────────

    def _show_load_screen(self):
        self.landing_frame.pack_forget()
        self.quiz_frame.pack_forget()
        self.load_frame.pack(fill="both", expand=True)

    def _show_landing_screen(self):
        self.load_frame.pack_forget()
        self.quiz_frame.pack_forget()
        self.landing_frame.pack(fill="both", expand=True)

    def _show_quiz_screen(self):
        self.load_frame.pack_forget()
        self.landing_frame.pack_forget()
        self.quiz_frame.pack(fill="both", expand=True)

    # ── CSV loading ───────────────────────────────────────────────────────────

    def _open_csv(self):
        path = filedialog.askopenfilename(
            title="Open human_dataset.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self._load_csv(path)

    def _open_save(self):
        path = filedialog.askopenfilename(
            title="Open Progress Save",
            filetypes=[("JSON Save files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        
        try:
            with open(path, "r") as f:
                state = json.load(f)
            
            self.csv_path = state.get("csv_path")
            self.rows = state.get("rows", [])
            self.current = state.get("current", 0)
            self.elapsed = state.get("elapsed", 0.0)
            
            raw_ans = state.get("answers", {})
            self.answers = {int(k): v for k, v in raw_ans.items()}
            
            raw_ca = state.get("can_answer", {})
            self.can_answer = {int(k): v for k, v in raw_ca.items()}
            
            raw_cr = state.get("clinically_relevant", {})
            self.clinically_relevant = {int(k): v for k, v in raw_cr.items()}
            
            raw_seq = state.get("necessary_sequences", {})
            self.necessary_sequences = {int(k): v for k, v in raw_seq.items()}
            
            raw_conf = state.get("confidence", {})
            self.confidence = {}
            for k, v in raw_conf.items():
                val = float(v)
                if val > 1.0:
                    val = val / 10.0  # legacy conversion
                self.confidence[int(k)] = val
            
            raw_com = state.get("comments", {})
            self.comments = {int(k): v for k, v in raw_com.items()}

            n_q = len(self.rows)
            n_acc = len(set([r.get("Deidentified_Accession_Number") for r in self.rows]))
            fname = os.path.basename(self.csv_path or "Unknown")

            self._landing_n_questions_lbl.config(text=str(n_q))
            self._landing_n_accessions_lbl.config(text=str(n_acc))
            self._landing_filename_lbl.config(text=fname, font=self.FONT_SMALL)
            
            # Format time
            m, s = divmod(int(self.elapsed), 60)
            h, m = divmod(m, 60)
            t_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"

            self.landing_subtitle.config(text=f"Resuming saved progress from: {os.path.basename(path)}\nTime spent so far: {t_str}")

            self._show_landing_screen()

        except Exception as e:
            self.load_err.config(text=f"Could not load save: {e}")
            self._show_load_screen()

    def _load_csv(self, path: str):
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self.load_err.config(text=f"Could not read file: {e}")
            self._show_load_screen()
            return

        required = {"Question", "Deidentified_Accession_Number"}
        missing = required - set(df.columns)
        if missing:
            self.load_err.config(text=f"Missing columns: {', '.join(missing)}")
            self._show_load_screen()
            return

        self.csv_path = path
        self.rows = df.to_dict("records")
        self.current = 0
        self.answers = {}
        self.can_answer = {}
        self.clinically_relevant = {}
        self.confidence = {}
        self.necessary_sequences = {}
        self.comments = {}

        # populate landing stats
        n_q = len(self.rows)
        n_acc = df["Deidentified_Accession_Number"].nunique()
        fname = os.path.basename(path)
        self._landing_n_questions_lbl.config(text=str(n_q))
        self._landing_n_accessions_lbl.config(text=str(n_acc))
        self._landing_filename_lbl.config(text=fname, font=self.FONT_SMALL)
        self.landing_subtitle.config(text=f"Loaded from: {path}")

        self._show_landing_screen()

    def _start_exam(self):
        # adjust start_time so that time.time() - start_time = self.elapsed
        self.start_time = time.time() - self.elapsed
        self._start_timer()
        self._show_quiz_screen()
        self._render_question(self.current)

    # ── timer ─────────────────────────────────────────────────────────────────

    def _start_timer(self):
        def tick():
            self.elapsed = time.time() - self.start_time
            self.timer_lbl.config(text=f"⏱  {format_elapsed(self.elapsed)}")
            self._timer_id = self.after(1000, tick)
        tick()

    def _stop_timer(self):
        if self._timer_id:
            self.after_cancel(self._timer_id)
            self._timer_id = None

    # ── question rendering ────────────────────────────────────────────────────

    def _render_question(self, idx: int):
        self.current = idx
        row = self.rows[idx]
        n = len(self.rows)

        # header
        self.accession_lbl.config(text=str(row.get("Deidentified_Accession_Number", "—")))
        self.progress_lbl.config(text=f"{idx + 1} of {n}")
        self.prog_bar["maximum"] = n
        self.prog_bar["value"]   = idx + 1

        # question stem
        stem, opts = parse_question(str(row.get("Question", "")))
        self.q_text.config(state="normal")
        self.q_text.delete("1.0", "end")
        self.q_text.insert("1.0", stem)
        # auto-resize height
        line_count = max(2, stem.count("\n") + len(stem) // 72 + 1)
        self.q_text.config(height=min(line_count, 8), state="disabled")

        # options — value is the full string e.g. "1) Answer text"
        self.selected_var.set(self.answers.get(idx, ""))
        for i, (frame, rb) in enumerate(self.option_frames):
            if i < len(opts):
                num, text = opts[i]
                full = f"{num}) {text}"
                rb.config(text=full, value=full, state="normal")
                frame.pack(fill="x", pady=(0, 8))
            else:
                frame.pack_forget()

        self._refresh_option_colors()
        self._update_hint()

        # restore confidence
        sv_dec = float(self.confidence.get(idx, 0.5))
        sv = int(round(sv_dec * 10))
        self.confidence_var.set(sv)
        self.conf_val_lbl.config(text=f"{sv * 10}%")
        if hasattr(self, 'conf_slider'):
            self.conf_slider.draw()

        # restore additional questions
        self.can_answer_var.set(self.can_answer.get(idx, ""))
        self.clinically_relevant_var.set(self.clinically_relevant.get(idx, ""))
        self.sequences_var.set(self.necessary_sequences.get(idx, ""))

        # restore comment
        self.comment_box.delete("1.0", "end")
        self.comment_box.insert("1.0", self.comments.get(idx, ""))

        # buttons
        self.prev_btn.config(state="normal" if idx > 0 else "disabled")
        self.next_btn.config(text="Finish" if idx == n - 1 else "Next →")

    def _refresh_option_colors(self):
        selected = self.selected_var.get()
        for frame, rb in self.option_frames:
            if rb.cget("value") == selected and selected:
                frame.config(highlightbackground=self.SEL_BD, bg=self.SEL_BG)
                rb.config(bg=self.SEL_BG)
            else:
                frame.config(highlightbackground=self.BORDER, bg=self.CARD)
                rb.config(bg=self.CARD)

    def _on_select(self):
        self.answers[self.current] = self.selected_var.get()
        self._refresh_option_colors()
        self._update_hint()

    def _on_extra_select(self):
        self.can_answer[self.current] = self.can_answer_var.get()
        self.clinically_relevant[self.current] = self.clinically_relevant_var.get()

    def _on_conf_select(self):
        val = int(self.confidence_var.get())
        self.confidence[self.current] = round(val / 10.0, 1)

    def _save_text_fields(self, event=None):
        self.comments[self.current] = self.comment_box.get("1.0", "end").strip()
        self.necessary_sequences[self.current] = self.sequences_var.get().strip()

    def _update_hint(self):
        sel = self.answers.get(self.current, "")
        self.hint_lbl.config(
            text=f"Selected: {sel}" if sel else "No answer selected",
            fg=self.ACC if sel else self.MUTED
        )

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        self._save_text_fields()
        if self.current > 0:
            self._render_question(self.current - 1)

    def _next(self):
        self._save_text_fields()
        if self.current < len(self.rows) - 1:
            self._render_question(self.current + 1)
        else:
            self._finish()

    def _submit_now(self):
        self._save_text_fields()
        unanswered = len(self.rows) - len(self.answers)
        msg = (
            f"Are you sure you want to submit?\n\n"
            f"Answered:   {len(self.answers)} of {len(self.rows)}\n"
            f"Unanswered: {unanswered}"
        )
        if messagebox.askyesno("Confirm submission", msg, icon="warning"):
            self._finish()

    def _pause_now(self):
        self._save_text_fields()
        self._stop_timer()
        self.elapsed = time.time() - self.start_time

        save_path = filedialog.asksaveasfilename(
            title="Save Progress",
            defaultextension=".json",
            initialfile="qa_progress_save.json",
            filetypes=[("JSON files", "*.json")]
        )

        if not save_path:
            self.start_time = time.time() - self.elapsed
            self._start_timer()
            return

        state = {
            "csv_path": self.csv_path,
            "rows": self.rows,
            "current": self.current,
            "elapsed": self.elapsed,
            "answers": self.answers,
            "can_answer": self.can_answer,
            "clinically_relevant": self.clinically_relevant,
            "confidence": self.confidence,
            "necessary_sequences": self.necessary_sequences,
            "comments": self.comments
        }

        try:
            with open(save_path, "w") as f:
                json.dump(state, f, indent=2)
            
            messagebox.showinfo("Paused", f"Progress successfully saved to:\n{save_path}\n\nYou can safely close the app.")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save progress:\n{e}")
            self.start_time = time.time() - self.elapsed
            self._start_timer()

    # ── finish & save ─────────────────────────────────────────────────────────

    def _on_closing(self):
        if not getattr(self, 'rows', None):
            self.destroy()
            return
            
        unanswered = len(self.rows) - len(self.answers)
        if unanswered == len(self.rows) and getattr(self, 'elapsed', 0) < 5:
            # Just loaded, no progress
            self.destroy()
            return
            
        res = messagebox.askyesnocancel("Wait!", "Do you want to save your progress before quitting?\n\nIf you select 'No', all unsaved answers will be lost.")
        if res is True:
            self._pause_now()
        elif res is False:
            self.destroy()

    def _finish(self):
        self._save_text_fields()
        self._stop_timer()
        total_elapsed = time.time() - self.start_time

        # build results
        records = []
        for i, row in enumerate(self.rows):
            records.append({
                "index":                         i + 1,
                "Deidentified_Accession_Number": row.get("Deidentified_Accession_Number", ""),
                "Question":                      row.get("Question", ""),
                "selected_answer":               self.answers.get(i, ""),
                "can_answer_from_image":         self.can_answer.get(i, ""),
                "clinically_relevant":           self.clinically_relevant.get(i, ""),
                "confidence_score":              self.confidence.get(i, ""),
                "necessary_sequences":           self.necessary_sequences.get(i, ""),
                "comments":                      self.comments.get(i, ""),
            })

        meta = {
            "completed_at":     datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds":  round(total_elapsed, 1),
            "elapsed_formatted": format_elapsed(total_elapsed),
            "total_questions":  len(self.rows),
            "answered":         len(self.answers),
            "unanswered":       len(self.rows) - len(self.answers),
        }

        save_dir = os.path.dirname(self.csv_path) if self.csv_path else os.getcwd()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path  = os.path.join(save_dir, f"qa_results_{timestamp}.csv")

        results_df = pd.DataFrame(records)
        results_df.to_csv(out_path, index=False)

        # also save a companion JSON with metadata
        json_path = out_path.replace(".csv", "_meta.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        messagebox.showinfo(
            "Saved",
            f"Results saved to:\n{out_path}\n\n"
            f"Metadata (time, counts):\n{json_path}\n\n"
            f"Time taken: {format_elapsed(total_elapsed)}\n"
            f"Answered: {meta['answered']} / {meta['total_questions']}"
        )
        self.destroy()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QAReviewer()
    app.mainloop()