# 📘 Parameterized LLM-Based Assessment Generation

## 🔹 Introduction

Before proceeding with the analysis of the following code files, it is important to briefly outline the basic steps required for their execution. All scripts are implemented in Python, therefore a suitable development environment such as PyCharm or Visual Studio Code is required. 

Additionally, the installation of the necessary libraries (e.g., OpenAI, PyPDF2, as well as libraries for mathematical computation and visualization) is required, along with a valid API key to access OpenAI services. Finally, input files (such as `.txt` or `.pdf`) should be located in the same directory as the code.

---

## 🔹 Project Description

This project presents a structured approach for generating assessment-oriented educational material using Large Language Models (LLMs).

Instead of relying on ad hoc prompting, a **parameterized pipeline** is applied, allowing controlled content generation through explicitly defined parameters such as:

- Number of questions  
- Difficulty level  
- Target audience  
- Domain  

In this way, the generated material is:

- Consistent  
- Repeatable  
- Pedagogically aligned  

LLMs are used as controlled generation tools rather than autonomous content creators, which is particularly important in technical domains such as Machine Learning.

---

## 🔹 Conceptual Workflow (Pipeline Overview)

The process followed in this project is not simply a sequence of execution steps, but a **parameterized pipeline for generating educational assessment material**.

Specifically, the system is based on the following stages:

1. **Parameter Specification**  
   The parameters (Q, A, CA, DL, TGA, DOM) are explicitly defined, determining the structure and difficulty of the assessment.

2. **Prompt Construction**  
   The parameters are embedded into a **structured prompt template**, which serves as the interface between the user and the LLM.  
   The prompt includes clear instructions, constraints, and output format requirements.

3. **Constrained Generation (LLM Execution)**  
   The LLM is used as a **conditional generator**, meaning it produces content strictly based on the defined constraints rather than acting as an autonomous creator.

4. **Output Structuring**  
   The generated material follows a predefined format:
   - Numbered questions  
   - Answer options  
   - One correct answer  
   - No explanations  

5. **Evaluation & Iteration**  
   The results are evaluated and, if necessary, refined through parameter adjustments or prompt improvements.

This process ensures:
- Control  
- Repeatability  
- Transparency  

---

> The following steps describe the practical implementation workflow of the proposed parameterized pipeline, guiding the user from environment setup to controlled assessment generation and evaluation using an LLM.

---

## 🔹 Usage Process

Follow the steps below:

### 1. Install Required Tools

- Python (version 3.10 or later)  
- IDE:
  - PyCharm (recommended)  
  - or Visual Studio Code  

---

### 2. Create a Virtual Environment (Optional)

```
python -m venv venv
```

**Activate it**:

- **macOS / Linux**
```
source venv/bin/activate
```

- **Windows**
```
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Configure API Key

- **macOS / Linux**
```
export API_KEY="your_api_key_here"
```

- **Windows**
```
set API_KEY=your_api_key_here
```

### 5. Define Parameters

Example:
```
Q = 10
A = 4
CA = 1
DL = 3
TGA = "university-level learners"
DOM = "Machine Learning Optimization (Steepest Descent)"
```

### 6. Run the Program
```
python main.py
```
### 7. Evaluate Results

- Check clarity of questions

- Verify correctness of answers

- Ensure appropriate difficulty level

- Confirm that there is exactly one correct answer

### 8. Optimization

- Adjust parameters

- Improve prompt structure

- Add domain-specific reference material

---

## 🔹 Code Description

### ✅ API_text.py

- Reads the `prompt.txt` file

- Sends data to the `gpt-4o-mini` model

- Receives the response

- Saves the output to `output.txt`

- Displays a success message

### ✅ API_PDF.py

- Reads `document.pdf`

- Extracts content using `PyPDF2`

- Generates multiple-choice questions in Greek

- Includes:

   - Multiple answer options

   - One correct answer

   - Difficulty categorization

- Saves output to `output.txt`

### ✅ Steepest_Descent.py

Implements the Steepest Descent algorithm.

**Inputs**:

- Initial point

- Function

- Learning rate

- Termination criteria

**Stops when**:

- Gradient becomes very small

- Changes are negligible

- Maximum iterations are exceeded

**Outputs**:

- Minimum point

- Function value

- Number of iterations

- 3D plots

- 2D contour plots

---

## 🔹 Example Sets

The repository includes two example assessment sets:

- `Set_1.pdf`

- `Set_2.pdf`

These demonstrate how the parameterized pipeline generates assessment material under controlled configurations.

---

## 🔹 Tools Used

- ChatGPT 🤖

- GitHub Copilot 💡

- DeepSeek 🔍

These tools were used for:

- Development

- Understanding

- Code optimization



## 🔹 Additional Notes

- The pipeline is model-agnostic and can be applied to different LLM backends for comparative evaluation.
- In technical domains, strict correctness and adherence to formal definitions are critical, as even minor inaccuracies may lead to misconceptions.
- Optional supplementary material (e.g., theory, notes, or definitions) can be provided to the model to improve conceptual grounding and output quality.
- The three code implementations developed in the present study are provided in both Greek and English. This dual-language approach was adopted to enhance accessibility and usability, enabling users with different linguistic backgrounds to more easily understand, modify, and apply the code.
