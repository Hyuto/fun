from data import Data

if __name__ == "__main__":
    data = Data()
    for data_ in ['train', 'test']:
        images = data._extract_image(data_)
        labels = data._extract_label(data_)
        print(images.shape)