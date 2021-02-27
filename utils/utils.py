import matplotlib.pyplot as plt

def show_image(im, title, *, cmap='gray'):
    # axes for debugging purposes
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(im, cmap=cmap)
    
