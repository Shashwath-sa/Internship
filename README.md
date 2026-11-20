# Speaker Identification (Streamlit)

Closed-set speaker identification over 20 LibriSpeech train-clean-100 speakers, using 1D CNNs on Log-STFT and MFCC features.

Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.

Needs `models/best_stft_model.pth` and `models/best_mfcc_model.pth` (from Notebook 3). Optional sample clips go in `examples/`.
