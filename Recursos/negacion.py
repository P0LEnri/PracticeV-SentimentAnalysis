
def negacion_pln(texto):
    palabras = texto.split()
    negaciones = set(["no", "sin", "ni", "nada", "nunca", "tampoco", "nadie", "jamas", "ninguno"])
    nueva_lista = []

    i = 0
    while i < len(palabras):
        palabra_actual = palabras[i].lower()
        if palabra_actual in negaciones:
            try:
                siguiente1 = palabras[i+1].lower()
                siguiente2 = palabras[i+2].lower()

                # Asegurarse de que no haya signos de puntuación antes o después de las palabras
                
                nueva_lista.append(f"{palabra_actual}_{siguiente1}")
                nueva_lista.append(f"{palabra_actual}_{siguiente2}")
                i += 2
            except IndexError:
                pass
        else:
            nueva_lista.append(palabras[i])
        i += 1

    return " ".join(nueva_lista)

# Ejemplo de uso:
texto_original = "No me gusta el café, pero no me importaría probarlo. ni me gusta el té. tampoco me gusta el chocolate. nunca me gustó el café. jamás me gustó el té."
texto_modificado = negacion_pln(texto_original)
print(texto_modificado)
