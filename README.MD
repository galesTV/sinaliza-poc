# Sinaliza — Protótipo de Reconhecimento de Configurações de Mão

O **Sinaliza** é uma prova de conceito desenvolvida em Python para detectar e classificar três configurações estáticas de mão utilizando visão computacional e aprendizado baseado em exemplos.

O projeto utiliza a webcam para detectar os pontos de referência da mão, extrair características geométricas e comparar a configuração identificada com um conjunto de amostras previamente cadastradas.

> **Importante:** este projeto não é um tradutor de Libras nem reconhece a língua como um todo. Seu objetivo é experimentar técnicas de visão computacional aplicadas ao reconhecimento de configurações estáticas de mão.

## Funcionalidades

- Detecção de mãos em tempo real.
- Identificação dos 21 landmarks da mão.
- Normalização das coordenadas.
- Extração de sete características geométricas.
- Cadastro de amostras em arquivo JSON.
- Classificação automática de três configurações.
- Visualização dos landmarks e características na webcam.

## Configurações reconhecidas

| Configuração | Identificador |
|---|---|
| Mão aberta | `aberta` |
| Punho fechado | `punho` |
| Pinça | `pinca` |

## Tecnologias utilizadas

- Python 3.13.7
- OpenCV
- MediaPipe
- NumPy
- Matplotlib

## Como funciona

O processamento acontece em cinco etapas:

1. **Captura:** a webcam fornece os frames utilizados pelo sistema.
2. **Detecção:** o MediaPipe identifica 21 pontos de referência da mão.
3. **Normalização:** as coordenadas são ajustadas em relação ao pulso e a uma distância de referência.
4. **Extração:** são calculadas sete características geométricas, incluindo distâncias e ângulos entre os dedos.
5. **Classificação:** as características são padronizadas e comparadas com as amostras cadastradas usando distância euclidiana.

## Estrutura do projeto

```text
sinaliza-poc/
├── main.py
├── classifier.py
├── feature_extractor.py
├── analyze_dataset.py
├── visualize_dataset.py
├── convert_dataset.py
├── test_environment.py
├── test_normalization.py
├── test_features.py
├── dataset.json
├── dataset.csv
├── hand_landmarker.task
├── requirements.txt
└── README.md
```

## Instalação

Clone o repositório:

```bash
git clone URL_DO_REPOSITORIO
cd sinaliza-poc
```

Crie e ative um ambiente virtual:

```bash
python -m venv venv
```

No Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Coloque o arquivo `hand_landmarker.task` na raiz do projeto.

## Execução

Para iniciar a aplicação:

```bash
python main.py
```

### Controles

| Tecla | Ação |
|---|---|
| `1` | Selecionar mão aberta para cadastro |
| `2` | Selecionar punho para cadastro |
| `3` | Selecionar pinça para cadastro |
| `S` | Salvar uma nova amostra |
| `Q` | Encerrar a aplicação |

A identificação automática acontece enquanto a webcam detecta uma mão.

## Resultados

O classificador foi avaliado utilizando validação Leave-One-Out, na qual cada amostra é testada utilizando um modelo treinado com as demais amostras.

**Resultado obtido:**

- Total de amostras: 100.
- Classificações corretas: 99.
- Acurácia observada: 99%.

### Matriz de confusão

| Real / Previsto | Aberta | Punho | Pinça |
|---|---:|---:|---:|
| Aberta | 34 | 0 | 0 |
| Punho | 0 | 32 | 1 |
| Pinça | 0 | 0 | 33 |

Esse resultado representa o desempenho observado no conjunto de dados utilizado. Ele não garante a mesma precisão em outras pessoas ou condições de captura.

## Limitações

- Reconhece somente três configurações estáticas previamente cadastradas.
- Não interpreta movimentos ou sequências de sinais.
- Não reconhece expressões faciais ou outros elementos não manuais.
- A classificação depende das características presentes no dataset.
- A validação realizada não substitui testes com pessoas diferentes e condições variadas.

## Objetivo educacional

O projeto foi desenvolvido como uma experiência prática de visão computacional, processamento de landmarks, extração de características e classificação utilizando Python.

A intenção é construir uma base técnica que permita explorar futuramente aplicações educacionais relacionadas à aprendizagem de Libras.

---

**Sinaliza — Uma experiência com visão computacional e reconhecimento de configurações de mão.**