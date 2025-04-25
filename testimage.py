from PIL import Image
import pytesseract

img_path = "pdf_images/page_5_img_0.png"
text = pytesseract.image_to_string(Image.open(img_path))
print(text)