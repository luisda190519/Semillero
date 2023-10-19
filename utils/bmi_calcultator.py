import os

def create_output_directory(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_directory = os.path.join("Salida", image_name)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    return output_directory

def BMI_calculator(bmi):
    if bmi < 16:
        return "Delgadez severa"
    elif bmi >= 16 and bmi <= 17:
        return "Delgadez moderada"
    elif bmi >= 17 and bmi <= 18.5:
        return "Delgadez leve"
    elif bmi >= 18.5 and bmi <= 25:
        return "Normal"
    elif bmi >= 25 and bmi <= 30:
        return "Sobrepeso"
    elif bmi >= 30 and bmi <= 35:
        return "Obesidad Clase I"
    elif bmi >= 35 and bmi <= 40:
        return "Obesidad Clase II"
    else:
        return "Obesidad Clase III"