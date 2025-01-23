# OCR Handwritten Sentence Recognition

Este projeto tem como objetivo reconhecer frases escritas à mão a partir de imagens e retornar a precisão da previsão. O modelo foi treinado para identificar textos manuscritos e fornecer uma avaliação precisa da frase extraída da imagem.


Link para download do IAM dataset utilizado: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database

## Funcionalidades

- **Reconhecimento de Texto:** O modelo reconhece frases manuscritas a partir de uma imagem.
- **Avaliação de Precisão:** O sistema retorna a imagem, a label que corresponde à frase correta e a predição do modelo, além da proporção de erros em relação ao número total de letras e palavras.

## Como Usar

1. Clone o repositório:
   ```bash
   git clone https://github.com/Fauser77/OCRSentece_Project.git
   cd OCRSentece_Project
2. Treine o modelo (opcional):
   ```bash
   python train.py
3. Valide a eficácia:
   ```bash
   python inference.py

## Observações

- Certifique-se de que o dataset esteja corretamente estruturado antes de executar o script de treinamento.
- ⚠️ **Aviso:** O script `train.py` pode consumir muitos recursos da máquina. Recomenda-se utilizá-lo em um ambiente com GPU.
- 🛠 Caso encontre problemas com o script `inference.py`, verifique se o diretório correto do modelo treinado está sendo passado. 
