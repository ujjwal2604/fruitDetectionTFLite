import tensorflow as tf
from tensorflow.keras.models import *
import numpy as np

class Fruit:
    
    def __init__(self, img = ''):
        self.img = img
        self.model = load_model('Fruits_360.h5')
        self.Label_dict = labels =  {'Apple Braeburn': 0,
             'Apple Golden': 1,
             'Apple Granny Smith': 2,
             'Apple Red': 3,
             'Apricot': 4,
             'Avocado': 5,
             'Avocado ripe': 6,
             'Banana': 7,
             'Banana Lady Finger': 8,
             'Banana Red': 9,
             'Cactus fruit': 10,
             'Cantaloupe 1': 11,
             'Cantaloupe 2': 12,
             'Carambula': 13,
             'Cherry 1': 14,
             'Cherry Wax Black': 15,
             'Cherry Wax Red': 16,
             'Cherry Wax Yellow': 17,
             'Chestnut': 18,
             'Clementine': 19,
             'Cocos': 20,
             'Dates': 21,
             'Grape Blue': 22,
             'Grape Pink': 23,
             'Grape White': 24,
             'Grapefruit Pink': 25,
             'Grapefruit White': 26,
             'Guava': 27,
             'Hazelnut': 28,
             'Huckleberry': 29,
             'Kaki': 30,
             'Kiwi': 31,
             'Kumquats': 32,
             'Lemon': 33,
             'Lemon Meyer': 34,
             'Limes': 35,
             'Lychee': 36,
             'Mandarine': 37,
             'Mango': 38,
             'Mangostan': 39,
             'Melon Piel de Sapo': 40,
             'Mulberry': 41,
             'Nectarine': 42,
             'Orange': 43,
             'Papaya': 44,
             'Passion Fruit': 45,
             'Peach': 46,
             'Peach 2': 47,
             'Peach Flat': 48,
             'Pear': 49,
             'Pear Kaiser': 50,
             'Pineapple': 51,
             'Pineapple Mini': 52,
             'Pitahaya Red': 53,
             'Plum': 54,
             'Plum 2': 55,
             'Plum 3': 56,
             'Pomegranate': 57,
             'Pomelo Sweetie': 58,
             'Rambutan': 59,
             'Raspberry': 60,
             'Redcurrant': 61,
             'Strawberry': 62,
             'Strawberry Wedge': 63,
             'Tomato 1': 64,
             'Tomato 2': 65,
             'Tomato 4': 66,
             'Tomato Cherry Red': 67,
             'Tomato Maroon': 68,
             'Walnut': 69}
        self.label = list(self.Label_dict.keys())
    
    def predict(self):
        result=self.model.predict(self.img)
        result_list = result[0].tolist()
        result_class = self.label[result_list.index(max(result_list))]
        return result_class
      
    def predictLite(self):
        model = "fruits.tflite"
        interpreter = tf.lite.Interpreter(model_path = model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_shape = input_details[0]['shape']
        interpreter.set_tensor(input_details[0]['index'], self.img)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result_list = output_data[0].tolist()
        result__final_class = self.label[result_list.index(max(result_list))]
        return result__final_class
        