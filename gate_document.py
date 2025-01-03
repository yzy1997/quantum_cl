from fpdf import FPDF

# Create a PDF class instance
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# Title for the document
pdf.set_font("Arial", style='B', size=16)
pdf.cell(200, 10, txt="Quantum Gate Types Overview", ln=True, align='C')
pdf.ln(10)

# Adding the table in text format to the PDF
pdf.set_font("Arial", size=12)

headers = [
    "Gate Type", "Gate Name", "Symbol/Matrix", "Description"
]

data = [
    ["Single-Qubit", "Hadamard (H)", "1/sqrt(2) [[1, 1], [1, -1]]", "Creates superposition"],
    ["Single-Qubit", "Pauli-X", "[[0, 1], [1, 0]]", "Acts like a classical NOT gate"],
    ["Single-Qubit", "Pauli-Y", "[[0, -i], [i, 0]]", "Combines X and Z effects"],
    ["Single-Qubit", "Pauli-Z", "[[1, 0], [0, -1]]", "Flips the phase"],
    ["Single-Qubit", "S Gate", "[[1, 0], [0, i]]", "Applies a phase of π/2"],
    ["Single-Qubit", "T Gate", "[[1, 0], [0, e^(iπ/4)]]", "Applies a phase of π/4"],
    ["Multi-Qubit", "CNOT", "[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]", "Conditional NOT"],
    ["Multi-Qubit", "Toffoli", "3x3 Matrix", "Controlled-controlled NOT"],
    ["Multi-Qubit", "SWAP", "3x3 Matrix", "Swaps two qubits"],
    ["Measurement", "Z Basis", "-", "Measures in standard basis"],
    ["Measurement", "X Basis", "-", "Measures in superposition basis"],
]

pdf.cell(0, 10, "Quantum Gate Table", ln=True)
pdf.ln(5)

for row in [headers] + data:
    pdf.cell(40, 10, txt=row[0], border=1)
    pdf.cell(50, 10, txt=row[1], border=1)
    pdf.cell(60, 10, txt=row[2], border=1)
    pdf.cell(40, 10, txt=row[3], border=1)
    pdf.ln()

# Save the PDF
file_path = "/mnt/data/Quantum_Gate_Table.pdf"
pdf.output(file_path)

file_path
