from PIL import Image
from collections import Counter



def concat_images_with_bbox(images, bboxs=None, arrangement=(3,3), scale=1.0, line_width=40,max_pixel=None):
  
    
    images = [Image.open(img).convert('RGB') for img in images]
    sizes = [image.size for image in images]


    size_counts = Counter(sizes)
    most_common_size = size_counts.most_common(1)[0][0]
    width, height = most_common_size

    
    images_resized = [img.resize(most_common_size) for img in images]

 
    if isinstance(arrangement, tuple) and len(arrangement) == 2:
        rows, columns = arrangement
    elif arrangement == 'horizontal':
        rows = 1
        columns = len(images_resized)
    elif arrangement == 'vertical':
        rows = len(images_resized)
        columns = 1
    else:
      
        rows = 1
        columns = len(images_resized)

    
    total_cells = rows * columns

    if len(images_resized) < total_cells:
   
        num_padding = total_cells - len(images_resized)
        blank_img = Image.new('RGB', most_common_size, color=(255,255,255))  
        images_resized.extend([blank_img]*num_padding)
    elif len(images_resized) > total_cells:
       
        images_resized = images_resized[:total_cells]

   
    total_width = columns * width + (columns - 1) * line_width  
    total_height = rows * height + (rows - 1) * line_width  
    new_img = Image.new('RGB', (total_width, total_height), color=(0,0,0))  

   
    for idx, img in enumerate(images_resized):
        row = idx // columns
        col = idx % columns
        x = col * (width + line_width)  # ==
        y = row * (height + line_width)  #
        new_img.paste(img, (x, y))

    
    if max_pixel is not None:
        total_pixels = total_width * total_height
        scale = (max_pixel / total_pixels) ** 0.5
        scaled_width = int(total_width * scale)
        scaled_height = int(total_height * scale)
        final_img = new_img.resize((scaled_width, scaled_height))
    else:
        if scale > 1:
            max_side = max(total_width, total_height)
            scale = scale / max_side
        scaled_width = int(total_width * scale)
        scaled_height = int(total_height * scale)
        final_img = new_img.resize((scaled_width, scaled_height))

    return final_img