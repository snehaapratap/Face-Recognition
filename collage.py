from PIL import Image
import os

def make_collage(
    folder_path='data/test_images',
    collage_path='data/group_photos/collage.jpg',
    rows=2,
    cols=3,  
    image_size=(160, 160)
):
    os.makedirs(os.path.dirname(collage_path), exist_ok=True)

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)[:rows * cols]

    if len(image_files) < rows * cols:
        rows = 1
        cols = len(image_files)
        print(f"⚠️ Not enough images for 2x3 grid, switching to {rows}x{cols}.")

    collage_width = cols * image_size[0]
    collage_height = rows * image_size[1]
    collage_image = Image.new('RGB', (collage_width, collage_height), color='white')

    for idx, img_name in enumerate(image_files):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = Image.open(img_path).resize(image_size)
            row, col = divmod(idx, cols)
            collage_image.paste(img, (col * image_size[0], row * image_size[1]))
        except Exception as e:
            print(f"⚠️ Error with {img_name}: {e}")

    collage_image.save(collage_path)
    print(f"✅ Collage saved to {collage_path}")

if __name__ == "__main__":
    make_collage()
