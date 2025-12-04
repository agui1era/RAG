from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

text = """1. Acerca de XYZ Corp.
XYZ Corp. fue fundada en el año 2010 en Valparaíso. La empresa se especializa en el desarrollo de soluciones tecnológicas para la industria pesquera.

2. Estadísticas clave (2023):
- Empleados: 120.
- Facturación anual: 15 millones de dólares.
- Presencia en: Chile, Perú, y Ecuador.
- Proyectos destacados: Plataforma AquaTech para la gestión inteligente de redes pesqueras.

3. Futuro de XYZ Corp.
La empresa planea expandir sus operaciones a México en 2025, con un enfoque en soluciones sostenibles para la pesca industrial."""

pdf_path = os.path.join("pdfs", "xyz_corp.pdf")
if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

c = canvas.Canvas(pdf_path, pagesize=letter)
width, height = letter
y = height - 50

for line in text.split('\n'):
    c.drawString(50, y, line)
    y -= 20

c.save()
print(f"Created {pdf_path}")
