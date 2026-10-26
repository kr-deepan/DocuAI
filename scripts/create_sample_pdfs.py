"""
Script to generate PDF versions of sample documents and a synthetic scanned PDF (image-rendered) for OCR testing.
"""
import os
from PIL import Image, ImageDraw, ImageFont

def generate_scanned_invoice_image(output_path: str):
    """Generates an image file containing text to simulate a scanned document for OCR testing."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    img = Image.new('RGB', (800, 1000), color=(250, 250, 245))
    d = ImageDraw.Draw(img)
    
    # Render simulated scanned invoice text
    text_lines = [
        "INVOICE # INV-2024-9981",
        "Date: July 10, 2024",
        "Vendor: Acme Cloud Solutions Inc.",
        "Bill To: Globex Corporation",
        "----------------------------------------",
        "Item Description           Qty    Total",
        "Managed Cloud Server        1   $5,000.00",
        "Premium SLA Tier Upgrade   1   $1,200.00",
        "----------------------------------------",
        "Subtotal:                       $6,200.00",
        "Tax (0%):                           $0.00",
        "Total Due:                      $6,200.00",
        "Payment Terms: Net-30 Days",
        "Note: Scan copy for auditing compliance."
    ]
    
    y = 50
    for line in text_lines:
        d.text((50, y), line, fill=(20, 20, 20))
        y += 40
        
    img.save(output_path)
    print(f"Generated scanned invoice image at: {output_path}")

if __name__ == "__main__":
    image_path = os.path.join("data", "sample_docs", "scanned_invoice_sample.png")
    generate_scanned_invoice_image(image_path)
