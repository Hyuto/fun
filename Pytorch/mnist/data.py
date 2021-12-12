import os, gzip, argparse, logging
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


class Data:
    """Data extractor

    Reference : https://stackoverflow.com/a/53570674"""

    CONFIG = {
        "dir": "./dataset",
        "image_size": 28,
        "train": {
            "images": "train-images-idx3-ubyte.gz",
            "labels": "train-labels-idx1-ubyte.gz",
            "n_items": 60000,
        },
        "test": {
            "images": "t10k-images-idx3-ubyte.gz",
            "labels": "t10k-labels-idx1-ubyte.gz",
            "n_items": 10000,
        },
    }

    def _extract_image(self, which):
        """Extract images"""
        path = os.path.join(self.CONFIG["dir"], self.CONFIG[which]["images"])
        f = gzip.open(path, "r")
        f.read(16)  # skip line

        buf = f.read(
            self.CONFIG["image_size"] * self.CONFIG["image_size"] * self.CONFIG[which]["n_items"]
        )
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(
            self.CONFIG[which]["n_items"], self.CONFIG["image_size"], self.CONFIG["image_size"], 1
        )
        return data

    def extract_prep_images(self, which):
        """Extract and preprocessing images dataset"""
        data = self._extract_image(which)
        # Normalize
        data = data / 255.0
        return data

    def _extract_label(self, which):
        """Extract labels"""
        path = os.path.join(self.CONFIG["dir"], self.CONFIG[which]["labels"])
        f = gzip.open(path, "r")
        f.read(8)  # skip line

        labels = np.asarray(
            [
                np.frombuffer(f.read(1), dtype=np.uint8).astype(np.int64)
                for i in range(self.CONFIG[which]["n_items"])
            ]
        ).astype(np.int32)

        return labels.flatten()


def get_rand_img(images, labels, n=10):
    """Select random image"""
    # Get random number with n size for index
    rand_num = np.random.randint(0, high=len(images), size=n, dtype=np.int32)
    # Select image based in the random number
    rand_images = images[rand_num]
    # Make a list of ImageTK
    img = [
        ImageTk.PhotoImage(image=Image.fromarray(np.asarray(x).squeeze()).resize((200, 200)))
        for x in rand_images
    ]
    return img, labels[rand_num]


class MainWindow:
    """Main window for showing random image"""

    def __init__(self, root, images, labels):
        self.parent = tk.Frame(root)
        self.canvas = tk.Canvas(self.parent, width=200, height=200)
        self.canvas.pack(expand=1)

        self.text = tk.StringVar()
        self.text_label = tk.Label(self.parent, textvariable=self.text)
        self.text_label.pack(expand=1)

        self.parent.pack(expand=1)

        self.images, self.labels = get_rand_img(images, labels)
        self.state = 0

        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw", image=self.images[self.state])
        self.text.set(f"Label : {self.labels[self.state]}")

        self.button = tk.Button(root, text="next", command=self.onClick)
        self.button.pack(expand=1)

    def onClick(self):
        """Change image when button clicked"""
        if self.state + 1 >= len(self.images):
            self.images, self.labels = get_rand_img(images, labels)
            self.state = 0
        else:
            self.state += 1
        self.canvas.itemconfig(self.canvas_img, image=self.images[self.state])
        self.text.set(f"Label : {self.labels[self.state]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data handling script")
    parser.add_argument("-rs", "--random_show", help="Randomly show images from data", type=str)

    args = parser.parse_args()

    if args.random_show:
        logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)
        logging.info(f"Showing randow image for {args.random_show} data")

        logging.info(f"Extracting {args.random_show} data")
        data = Data()
        images = data._extract_image(args.random_show)
        labels = data._extract_label(args.random_show)

        logging.info("Opening randow image GUI")
        root = tk.Tk()
        root.title("Random Image")
        root.geometry("270x280")
        MainWindow(root, images, labels)
        root.mainloop()
        logging.info("Closing GUI")
