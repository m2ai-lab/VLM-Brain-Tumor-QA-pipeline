# Brain MRI QA — Human Evaluation

## Directory Structure

```
brain_vlm_human_ds/
├── README.md
├── human_dataset.csv       # 250 questions across 16 studies
├── human_qa_gui.py         # Tkinter GUI to administer the exam
├── requirements.txt        # Python dependencies (pandas)
└── <Accession_Number>/     # One subdirectory per study
    └── ...                 # DICOM images for that scan
```

Each subdirectory is named after its accession number and contains all DICOM images for that scan. The accession number displayed in the top-left corner of the GUI corresponds directly to one of these folders.

---

## Setup

Install the single dependency:

```bash
pip install -r requirements.txt
```

---

## Running the Exam

```bash
python human_qa_gui.py
```

Place `human_dataset.csv` in the same directory as `human_qa_gui.py` and it will be detected automatically. If it is not found, the GUI will prompt you to browse for it manually.

---

## Exam Instructions

1. **Launch** — Run the command above. A landing page will display the number of questions and studies loaded.

2. **Start** — Press **Start exam**. The timer begins at this point.

3. **Viewing images** — The accession number for each question is shown in the top-left corner of the interface. Open the folder with the matching name inside `/mnt/fac/CX000019_DS1/brain_vlm_human_ds/` using your preferred DICOM viewing application to inspect the corresponding scan.

4. **Answering** — Read the question and click your selected answer choice. All answer options are displayed as buttons; your current selection is highlighted.

5. **Comments** — Use the comments box at the bottom of each question to note anything relevant — uncertainty, image quality issues, ambiguous answer choices, or anything else worth recording.

6. **Navigation** — Click **Next** to advance to the next question, or **Previous** to go back and revise a response. You can revisit any question at any time.

7. **Finishing** — On the last question, click **Finish** to submit. At any point during the exam you may also click **Submit now** to end early — you will be asked to confirm before submitting.

---

## Output

Results are saved automatically to the same directory as `human_qa_gui.py` when the exam is completed.

Two files are written:

| File | Contents |
|------|----------|
| `qa_results_YYYYMMDD_HHMMSS.csv` | One row per question: accession number, question text, selected answer, and comments |
| `qa_results_YYYYMMDD_HHMMSS_meta.json` | Session summary: completion timestamp, total time elapsed, and answered/unanswered counts |