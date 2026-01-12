import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageTk

model = load_model("best_model.h5")
class_names = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets",
    "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad",
    "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings",
    "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee",
    "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots",
    "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast",
    "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog",
    "huevos_rancheros", "hummus", "ice_cream", "lasagna", "lobster_bisque", "lobster_roll_sandwich",
    "macaroni_and_cheese", "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck", "pho", "pizza", "pork_chop", "poutine",
    "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]

def predict_image():
    file_path = filedialog.askopenfilename(title="Wybierz obraz",
                                           filetypes=[("JPG files", "*.jpg"), ("PNG files", "*.png")])

    if file_path:
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        percentages = [(class_names[i], predictions[i] * 100) for i in range(len(class_names))]
        percentages.sort(key=lambda x: x[1], reverse=True)
        top_5 = percentages[:5]

        result_text = "\n".join([f"{cls}: {prob:.2f}%" for cls, prob in top_5])
        result_label.config(text=result_text)

        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

root = tk.Tk()
root.title("Rozpoznawanie Jedzenia")

main_frame = tk.Frame(root)
main_frame.pack(padx=10, pady=10)

image_frame = tk.Frame(main_frame)
image_frame.grid(row=0, column=0, padx=10, pady=10)

results_frame = tk.Frame(main_frame)
results_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")

upload_button = tk.Button(root, text="Wczytaj Obraz", command=predict_image)
upload_button.pack(pady=10)

image_label = tk.Label(image_frame)
image_label.pack()

result_label = tk.Label(results_frame, text="Predykcja: Brak", font=("Arial", 12), justify="left")
result_label.pack()

root.mainloop()
