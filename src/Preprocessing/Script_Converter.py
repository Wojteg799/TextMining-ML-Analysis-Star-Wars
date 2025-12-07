import docx
import re
import os

def czy_nazwa_postaci(tekst):
    return tekst.isupper() and len(tekst.strip()) > 1
def konwertuj_docx_do_txt(sciezka_docx, sciezka_wyjsciowa):

    dokument = docx.Document(sciezka_docx)

    przetworzone_linie = []
    aktualna_postac = None
    for paragraf in dokument.paragraphs:
        tekst = paragraf.text.strip()
        if not tekst: 
            continue
            
        if paragraf.style.name.startswith('Heading') and czy_nazwa_postaci(tekst):
            if aktualna_postac: 
                przetworzone_linie.append("[KONIEC_DIALOGU]")
            aktualna_postac = tekst
            przetworzone_linie.append(f"{tekst} [POCZATEK_DIALOGU]")
        else:
            if aktualna_postac: 
                przetworzone_linie.append(tekst)
            else: 
                przetworzone_linie.append(f"[OPIS_SCENY] {tekst}")
    
    if aktualna_postac:
        przetworzone_linie.append("[KONIEC_DIALOGU]")
    
    with open(sciezka_wyjsciowa, 'w', encoding='utf-8') as plik:
        plik.write('\n'.join(przetworzone_linie))

def main():
    pliki_docx = [
        'star-wars-episode-iv-a-new-hope-1977.docx',
        'star-wars-episode-v-the-empire-strikes-back-1980.docx',
        'star-wars-episode-vi-return-of-the-jedi-1983.docx'
    ]
    
    for plik_docx in pliki_docx:
        if os.path.exists(plik_docx):
            plik_wyjsciowy = plik_docx.replace('.docx', '_format.txt')
            print(f"Konwertuję {plik_docx} na {plik_wyjsciowy}...")
            konwertuj_docx_do_txt(plik_docx, plik_wyjsciowy)
            print(f"Zakończono konwersję {plik_docx}")
        else:
            print(f"Nie znaleziono pliku {plik_docx}")

if __name__ == "__main__":
    main() 