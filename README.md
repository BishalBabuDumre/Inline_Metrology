### Prerequisites
python, git
## How to Run
Go to a folder in your terminal or command prompt in your computer.
### 1. Clone the repository:<br>
   git clone https://github.com/BishalBabuDumre/Inline_Metrology.git<br>
   cd Inline_Metrology

### 2. Create and activate a virtual environment:
   #### On macOS / Linux:
   python3 -m venv venv<br>
   source venv/bin/activate

   #### On Windows (Command Prompt / PowerShell):
   python -m venv venv<br>
   venv\Scripts\activate

### 3. Install dependencies:
   pip install -r requirements.txt

### 4. Run the script:
   cd code<br>
   cd initial_exploration<br>
   python initial.py (Does initial analysis on the data determining data types, duplicates, and missing values and prints it on the screen)<br>
   python graph.py (Creates graphs related to OES bins, voltage & current sweeps, and IR pixels inside main/data/results/initial_exploration) <br>
   cd ../machine_learning<br>
   python predict.py (Trains Machine Learning model where you can observe loss values being minimized on your screen)
