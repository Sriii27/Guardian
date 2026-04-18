from PIL import Image, ImageDraw, ImageFont
import os

# Create logo
img = Image.new('RGBA', (200, 200), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# Outer circle
draw.ellipse([10, 10, 190, 190], outline='#ff6b35', width=4)
draw.ellipse([20, 20, 180, 180], outline='#ff6b35', width=2)

# Shield shape
shield = [(100, 30), (170, 60), (170, 120), (100, 170), (30, 120), (30, 60)]
draw.polygon(shield, outline='#64ffda', fill='#0d1b2a', width=3)

# G letter in center
draw.ellipse([65, 65, 135, 135], outline='#ff6b35', width=3)
draw.rectangle([100, 85, 130, 100], fill='#ff6b35')
draw.rectangle([100, 85, 115, 115], fill='#ff6b35')

img.save('logo.png')
print("Logo created!")