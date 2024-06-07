import tkinter as tk
from recognition import EmotionRecognitionApp

def start_recognition():
    # Cerrar la ventana de bienvenida
    welcome_root.destroy()
    
    # Crear la ventana principal para la detección de emociones
    recognition_root = tk.Tk()
    app = EmotionRecognitionApp(recognition_root, show_welcome)
    recognition_root.mainloop()

def show_welcome():
    # Crear la ventana de bienvenida
    global welcome_root
    welcome_root = tk.Tk()
    welcome_root.title("Detección de Emociones")

    # Crear un marco para la bienvenida
    welcome_frame = tk.Frame(welcome_root)
    welcome_frame.pack(pady=20)

    welcome_label = tk.Label(welcome_frame, text="¡Bienvenido al programa de Detección de Emociones!", font=("Helvetica", 16))
    welcome_label.pack()

    # Crear un marco para el botón de inicio
    button_frame = tk.Frame(welcome_root)
    button_frame.pack(pady=20)

    start_button = tk.Button(button_frame, text="Iniciar Reconocimiento Facial", command=start_recognition, font=("Helvetica", 14))
    start_button.pack()

    # Ejecutar el bucle principal de la interfaz gráfica de bienvenida
    welcome_root.mainloop()

# Mostrar la ventana de bienvenida
show_welcome()
