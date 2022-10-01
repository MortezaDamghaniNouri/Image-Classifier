from keras.models import load_model
from PIL import Image
import numpy as np


# This function recognizes the output class of the CNN
def output_class_recognizer(input_list):
    if int(input_list[0]) == 1:
        print("The input image is: Airplane")
        return 0
    if int(input_list[1]) == 1:
        print("The input image is: Automobile")
        return 0
    if int(input_list[2]) == 1:
        print("The input image is: Bird")
        return 0
    if int(input_list[3]) == 1:
        print("The input image is: Cat")
        return 0
    if int(input_list[4]) == 1:
        print("The input image is: Deer")
        return 0
    if int(input_list[5]) == 1:
        print("The input image is: Dog")
        return 0
    if int(input_list[6]) == 1:
        print("The input image is: Frog")
        return 0
    if int(input_list[7]) == 1:
        print("The input image is: Horse")
        return 0
    if int(input_list[8]) == 1:
        print("The input image is: Ship")
        return 0
    if int(input_list[9]) == 1:
        print("The input image is: Truck")
        return 0


model = load_model('model3.h5')
results = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
input_image_name = input("Enter the full name of the input image(like image1.jpg): ")
main_input_image = Image.open(input_image_name)
input_image = main_input_image.resize((32, 32))
print(input_image)
resized_input_image = input_image
input_image = np.expand_dims(input_image, axis=0)
input_image = np.array(input_image)
# print(input_image)





output_class_recognizer(model.predict([input_image])[0])
main_input_image.show()
resized_input_image.show()










