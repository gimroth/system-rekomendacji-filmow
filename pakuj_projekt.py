import os

# Nazwa pliku wyjściowego
output_file = 'caly_projekt.txt'

# Foldery i pliki do całkowitego pominięcia
ignore_list = {
    '.git', '__pycache__', 'venv', '.idea', '.vscode', 
    'node_modules', '.DS_Store', 'pakuj_projekt.py', 
    output_file, 'db.sqlite3', '*.pyc', 
    # UWAGA: Usunąłem stąd pliki .csv, żeby skrypt do nich zajrzał,
    # ale logicznie utniemy je do 10 linii w pętli poniżej.
}

def is_text_file(filename):
    # Rozszerzenia, które chcemy czytać
    valid_extensions = [
        '.py', '.js', '.html', '.css', '.txt', '.md', 
        '.json', '.xml', '.csv', '.sql', '.java', '.c', '.cpp'
    ]
    return any(filename.endswith(ext) for ext in valid_extensions) or filename == 'Dockerfile' or 'requirements' in filename

with open(output_file, 'w', encoding='utf-8') as outfile:
    for root, dirs, files in os.walk("."):
        # Pomijanie folderów z listy ignore
        dirs[:] = [d for d in dirs if d not in ignore_list]
        
        for file in files:
            if file in ignore_list:
                continue
            
            if not is_text_file(file):
                continue

            file_path = os.path.join(root, file)
            
            try:
                # Otwieramy plik do odczytu
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                    
                    # Logika dla plików CSV (tylko 10 linii)
                    if file.endswith('.csv'):
                        lines = []
                        for i, line in enumerate(infile):
                            if i < 10:
                                lines.append(line)
                            else:
                                lines.append(f"\n... [Reszta pliku CSV pominięta. Oryginał ma więcej linii.] ...\n")
                                break
                        content = "".join(lines)
                        print(f"Dodano (snippet): {file_path}")
                    
                    # Logika dla reszty plików (całość)
                    else:
                        content = infile.read()
                        print(f"Dodano (całość): {file_path}")
                    
                    # Zapis do pliku zbiorczego
                    outfile.write(f"\n{'='*50}\n")
                    outfile.write(f"PLIK: {file_path}\n")
                    outfile.write(f"{'='*50}\n")
                    outfile.write(content + "\n")
                    
            except Exception as e:
                print(f"Pominięto {file_path} (błąd odczytu: {e})")

print(f"\nGotowe! Cały projekt jest w pliku: {output_file}")