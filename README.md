Loan Approval System

A simple Python CLI app that:
	1.	Loads (or generates) a dataset of applicants with a CivilScore feature
	2.	Trains a Random Forest to predict loan approval
	3.	Evaluates performance
	4.	Lets you interactively predict on new applicants

Setup
	1.	git clone <repo-url> && cd loan_approval_system
	2.	python3 -m venv .venv && source .venv/bin/activate
	3.	pip install -r requirements.txt

Usage

python loan_approval.py

Choose:
	•	1 to preprocess & train
	•	2 to run metrics (classification report & confusion matrix)
	•	3 to enter applicant details manually and see Approved/Rejected
	•	4 to Exit

If no data/loan_data.csv exists, it’ll auto-generate 1,000 synthetic records.

⸻

Example Cases

Below are some example input scenarios and the expected result:

Case 1: Approved

Income: 75000
Age: 35
CivilScore: 700
LoanAmount: 150000
CreditHistory: 1
→ ✅ Approved

Case 2: Rejected (low CivilScore)

Income: 60000
Age: 40
CivilScore: 550
LoanAmount: 120000
CreditHistory: 1
→ ❌ Rejected

Case 3: Rejected (no Credit History)

Income: 80000
Age: 28
CivilScore: 720
LoanAmount: 200000
CreditHistory: 0
→ ❌ Rejected

Case 4: Rejected (low score and no history)

Income: 45000
Age: 50
CivilScore: 580
LoanAmount: 100000
CreditHistory: 0
→ ❌ Rejected


⸻"# loan_approval_prediction" 
