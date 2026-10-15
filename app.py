from pathlib import Path
from datetime import datetime
import hashlib
import io
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent
CLASSIFIER_PATH = ROOT / 'runs/classify/train/weights/best.pt'
SAMPLE_PATH = ROOT / 'images/test.png'
DETECTOR_CANDIDATES = [
    ROOT / 'runs/detect/runs/detect/waste_v1-2/weights/best.pt',
    ROOT / 'runs/detect/waste_v1-2/weights/best.pt',
    ROOT / 'runs/detect/waste_v1/weights/best.pt',
]
DETECTOR_PATH = next((p for p in DETECTOR_CANDIDATES if p.is_file()), None)
if DETECTOR_PATH is None:
    matches = sorted((ROOT / 'runs/detect').glob('**/weights/best.pt')) if (ROOT / 'runs/detect').exists() else []
    DETECTOR_PATH = matches[-1] if matches else None
CLASSIFICATION_CLASSES = ['plastic', 'glass', 'metal', 'paper', 'cardboard', 'trash']
DETECTION_CLASSES = ['glass', 'metal', 'paper', 'plastic', 'residual']
TIPS = {
    'plastic': 'Check the recycling symbol and your local collection rules. Empty and rinse accepted containers.',
    'glass': 'Empty and rinse glass bottles and jars. Check local rules for lids and broken glass.',
    'metal': 'Empty and rinse accepted metal cans and tins. Check your council guidance for other metals.',
    'paper': 'Keep recyclable paper clean and dry. Greasy or heavily contaminated paper may not be accepted.',
    'cardboard': 'Flatten clean, dry cardboard. Remove food contamination before recycling.',
    'trash': 'Check whether any parts can be reused or recycled before using general waste.',
    'residual': 'This is the detector dataset’s residual category; check local guidance before disposal.',
}
st.set_page_config(page_title='EcoVision AI | Smart Waste Analysis', page_icon='♻️', layout='wide')
st.markdown('''
<style>
.block-container {padding-top: 1.7rem; max-width: 1350px;}
h1, h2, h3 {letter-spacing: -.02em;}
[data-testid="stMetric"] {background: rgba(128,128,128,.065); padding: 15px; border: 1px solid rgba(128,128,128,.15); border-radius: 13px;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.15);}
</style>
''', unsafe_allow_html=True)
@st.cache_resource(show_spinner='Loading YOLO models...')
def load_yolo(path: str):
    return YOLO(path)
def open_image(source):
    with Image.open(source) as original:
        return ImageOps.exif_transpose(original).convert('RGB')
def image_signature(image, mode, threshold, source):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return hashlib.sha256(buffer.getvalue() + f'{mode}|{threshold}|{source}'.encode()).hexdigest()
def store_history(records, signature):
    if st.session_state.get('last_signature') == signature:
        return
    st.session_state.history.extend(records)
    st.session_state.last_signature = signature
def model_prediction(image, mode, classifier, detector, threshold):
    if mode == 'Whole-image classification':
        if classifier is None:
            raise RuntimeError(f'Classification weights not found: {CLASSIFIER_PATH}')
        result = classifier.predict(source=image, verbose=False)[0]
        if result.probs is None:
            raise RuntimeError('Selected weights are not an image classification model.')
        class_id = int(result.probs.top1)
        label = str(result.names[class_id]).lower()
        score = float(result.probs.top1conf)
        return image, [{'Category': label, 'Confidence': round(score * 100, 2)}]
    if detector is None:
        raise RuntimeError('Object detection weights not found. Check the trained detector path in the sidebar.')
    result = detector.predict(source=image, conf=threshold, imgsz=416, verbose=False)[0]
    annotated = Image.fromarray(result.plot()[:, :, ::-1])
    records = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            records.append({
                'Category': str(result.names[class_id]).lower(),
                'Confidence': round(float(box.conf.item()) * 100, 2),
            })
    return annotated, records
def analyze_image(image, source_name, mode, classifier, detector, threshold, source_key):
    signature = image_signature(image, mode, threshold, source_key)
    with st.spinner('Analyzing waste image...'):
        annotated, predictions = model_prediction(image, mode, classifier, detector, threshold)
    left, right = st.columns(2, gap='large')
    with left:
        st.image(image, caption='Original image', use_container_width=True)
    with right:
        st.image(annotated, caption='Detection results' if mode.startswith('Multiple') else 'Analyzed image', use_container_width=True)
    if not predictions:
        st.warning('No objects detected at the current confidence threshold. Try lowering the threshold or using a clearer image.')
        return
    if mode == 'Whole-image classification':
        label = predictions[0]['Category']
        score = predictions[0]['Confidence']
        st.success(f'Predicted category: {label.upper()}')
        st.metric('Prediction confidence', f'{score:.1f}%')
        st.progress(min(max(score / 100, 0.0), 1.0))
        st.info('This model predicts one category for the entire image, even if several objects are present.')
    else:
        st.success(f'Detected {len(predictions)} waste object(s)')
        summary = pd.DataFrame(predictions)
        counts = summary['Category'].value_counts()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(counts)
    st.subheader('♻️ Disposal guidance')
    for label in dict.fromkeys(p['Category'] for p in predictions):
        st.write(f'**{label.title()}:** {TIPS.get(label, "Check local disposal guidance.")}')
    st.caption('Disposal tips are general guidance, not verified local recycling rules.')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = [
        {
            'Time': now,
            'Source': source_name,
            'Mode': mode,
            'Category': p['Category'],
            'Confidence': p['Confidence'],
        }
        for p in predictions
        ]
    store_history(rows, signature)
if 'history' not in st.session_state:
    st.session_state.history = []
with st.sidebar:
    st.title('♻️ EcoVision AI')
    st.caption('Smart Waste Intelligence Platform')
    st.divider()
    page = st.radio(
        'Navigation',
        ['🏠 Overview', '🔎 Waste Analysis', '📷 Camera', '📊 Analytics', '🧠 Model Performance', 'ℹ️ About'],
        label_visibility='collapsed',
    )
    st.divider()
    st.markdown('**Model status**')
    st.write(('🟢' if CLASSIFIER_PATH.is_file() else '🔴') + ' Classification model')
    st.write(('🟢' if DETECTOR_PATH else '🔴') + ' Detection model')
    st.caption('Classification: 6 categories · Detection: 5 categories')
st.title('♻️ Smart Waste Classification & Detection')
st.caption('EcoVision AI · Automated Waste Analysis Using YOLOv8 Deep Learning')
st.divider()
classifier = None
if CLASSIFIER_PATH.is_file():
    try:
        classifier = load_yolo(str(CLASSIFIER_PATH))
    except Exception as exc:
        st.warning(f'Unable to load classification model: {exc}')
detector = None
if DETECTOR_PATH is not None:
    try:
         detector = load_yolo(str(DETECTOR_PATH))
    except Exception as exc:
        st.warning(f'Unable to load detection model: {exc}')
if page == '🏠 Overview':
    st.subheader('Project Overview')
    a, b, c, d = st.columns(4)
    a.metric('Classification categories', '6')
    b.metric('Validation top-1 accuracy', '91.3%')
    c.metric('Detection categories', '5')
    d.metric('Detection mAP@50', '81.6%', help='Epoch 10 validation result; not necessarily the best checkpoint evaluation.')
    st.divider()
    st.subheader('What this application can do')
    left, right = st.columns(2)
    with left:
        st.markdown('**🖼️ Whole-image classification**')
        st.write('Predicts one label: plastic, glass, metal, paper, cardboard or trash.')
        st.markdown('**🔎 Multiple-object detection**')
        st.write('Locates glass, metal, paper, plastic and residual waste using bounding boxes.')
    with right:
        st.markdown('**📷 Camera snapshots**')
        st.write('Capture a photo in your browser and analyze it.')
        st.markdown('**📊 Session analytics**')
        st.write('View prediction history, charts and downloadable CSV results.')
    st.info('For images containing several waste objects, use Multiple-object detection on the Waste Analysis page.')
    if SAMPLE_PATH.is_file():
        st.subheader('Project sample image')
        st.image(str(SAMPLE_PATH), width=520)
elif page == '🔎 Waste Analysis':
    st.subheader('Analyze a Waste Image')
    mode = st.radio('Analysis mode', ['Multiple-object detection', 'Whole-image classification'], horizontal=True)
    threshold = st.slider('Detection confidence threshold', 0.05, 0.95, 0.25, 0.05, disabled=(mode != 'Multiple-object detection'))
    uploaded = st.file_uploader('Upload a JPG or PNG image', type=['jpg', 'jpeg', 'png'])
    use_sample = st.checkbox('Use project sample image instead', disabled=not SAMPLE_PATH.is_file())
    try:
        if use_sample:
            analyze_image(open_image(SAMPLE_PATH), 'Sample image', mode, classifier, detector, threshold, 'sample')
        elif uploaded is not None:
            analyze_image(open_image(uploaded), uploaded.name, mode, classifier, detector, threshold, uploaded.name)
        else:
            st.info('Upload an image or select the sample image to begin.')
    except Exception as exc:
        st.error(f'Image processing failed: {exc}')
elif page == '📷 Camera':
    st.subheader('Camera Waste Analysis')
    st.caption('This feature captures still photos. Continuous video requires a separate streaming integration.')
    mode = st.radio('Camera analysis mode', ['Multiple-object detection', 'Whole-image classification'], horizontal=True)
    threshold = st.slider('Detection confidence threshold', 0.05, 0.95, 0.25, 0.05, disabled=(mode != 'Multiple-object detection'))
    camera = st.camera_input('Take a waste photo')
    if camera is not None:
        try:
            image = open_image(camera)
            analyze_image(image, 'Camera', mode, classifier, detector, threshold, 'camera')
        except Exception as exc:
            st.error(f'Camera image processing failed: {exc}')
elif page == '📊 Analytics':
    st.subheader('Waste Detection Analytics')
    st.caption('These statistics reflect this Streamlit session, not a permanent database.')
    history = st.session_state.history
    if not history:
        st.info('No predictions recorded yet. Analyze an image or camera photo first.')
    else:
        df = pd.DataFrame(history)
        counts = df['Category'].value_counts().rename_axis('Category').reset_index(name='Count')
        a, b, c = st.columns(3)
        a.metric('Predictions recorded', len(df))
        b.metric('Distinct categories', df['Category'].nunique())
        c.metric('Average confidence', f"{df['Confidence'].mean():.1f}%")
        left, right = st.columns(2)
        with left:
            st.plotly_chart(px.bar(counts, x='Category', y='Count', title='Waste categories'), use_container_width=True)
        with right:
            st.plotly_chart(px.pie(counts, names='Category', values='Count', hole=0.5, title='Category distribution'), use_container_width=True)
        st.subheader('Prediction history')
        st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
        st.download_button('⬇️ Download results as CSV', data=df.to_csv(index=False).encode('utf-8'), file_name='waste_predictions.csv', mime='text/csv')
        if st.button('Clear session history'):
            st.session_state.history = []
            st.session_state.pop('last_signature', None)
            st.rerun()
elif page == '🧠 Model Performance':
    st.subheader('Model Performance')
    st.markdown('**Whole-image classification · YOLOv8n-cls**')
    st.metric('Validation top-1 accuracy', '91.3%')
    st.write('Classes: ' + ', '.join(CLASSIFICATION_CLASSES))
    st.code(str(CLASSIFIER_PATH.relative_to(ROOT)))
    st.divider()
    st.markdown('**Object detection · YOLOv8n**')
    a, b, c, d = st.columns(4)
    a.metric('Epoch 10 precision', '78.1%')
    b.metric('Epoch 10 recall', '79.1%')
    c.metric('Epoch 10 mAP@50', '81.6%')
    d.metric('Epoch 10 mAP@50–95', '59.4%')
    st.caption('These are recorded epoch-10 training-validation metrics, not a new independent test-set evaluation.')
    st.write('Classes: ' + ', '.join(DETECTION_CLASSES))
    if DETECTOR_PATH:
        st.code(str(DETECTOR_PATH.relative_to(ROOT)))
        results_path = DETECTOR_PATH.parent.parent / 'results.csv'
        if results_path.is_file():
            try:
                results_df = pd.read_csv(results_path)
                results_df.columns = results_df.columns.str.strip()
                if {'epoch', 'metrics/mAP50(B)'}.issubset(results_df.columns):
                    st.plotly_chart(px.line(results_df, x='epoch', y='metrics/mAP50(B)', markers=True, title='Detection validation mAP@50 by epoch'), use_container_width=True)
            except Exception as exc:
                st.warning(f'Could not read training history: {exc}')
    else:
        st.warning('Detection model weights not found.')
else:
    st.subheader('About the Project')
    st.write('**Project:** Automated Real-Time Waste Classification System Using Deep Learning')
    st.write('**Technology:** Python, Streamlit, Ultralytics YOLOv8, PyTorch, Pillow, Pandas and Plotly.')
    st.write('**Classification dataset:** Six-category garbage image dataset.')
    st.write('**Detection dataset:** Roboflow Waste Detection v5 with YOLOv8 bounding-box annotations.')
    st.write('**Limitations:** Camera captures snapshots, not continuous video. Detection supports five classes; classification supports six.')
    st.write('**Note:** Model predictions and disposal suggestions can be wrong. Follow your local recycling guidance.')
st.divider()
st.caption('EcoVision AI · Waste Classification & Detection · Built with Streamlit and YOLOv8')
