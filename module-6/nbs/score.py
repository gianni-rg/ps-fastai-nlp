from fastai.text import *
from azureml.core.model import Model
from html.parser import HTMLParser

class HTMLTextExtractor(html.parser.HTMLParser):
    def __init__(self):
        super(HTMLTextExtractor, self).__init__()
        self.result = [ ]

    def handle_data(self, d):
        self.result.append(d)

    def get_text(self):
        return ''.join(self.result)
    
    def error(self, message):
        return

def html_to_text(html):
    s = HTMLTextExtractor()    
    try:
        s.feed(html)
        return s.get_text()
    except:
        return html

def custom_tagstrip(x:str) -> str:
    "Remove all html tags in `x`."
    return html_to_text(x)

def load_model(classifier_filename):
    """Load the classifier and related metadata"""
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if torch.cuda.is_available():
        print('USING CUDA-GPU')
    else:
        print('USING CPU')
    
    state = torch.load(Path(classifier_filename).open('rb'), map_location=device)
    
    if set(state.keys()) == {'model', 'model_params', 'vocab', 'classes'}:
        model_state = state['model']
        model_params = state['model_params']
        itos = state['vocab']
        classes = state['classes']
    else:
        raise RuntimeError("Invalid model provided.")
        
    # Turn it into a string to int mapping (which is what we need)
    stoi = collections.defaultdict(lambda:0, {str(v):int(k) for k,v in enumerate(itos)})
    
    # Get model reference from parameters (even if they are not used at runtime)
    model = get_rnn_classifier(bptt=model_params['bptt'],
                               max_seq=model_params['max_len'],
                               vocab_sz=model_params['vocab_size'], 
                               emb_sz=model_params['emb_sz'],
                               n_hid=model_params['nh'],
                               n_layers=model_params['nl'],
                               pad_token=model_params['pad_token'],
                               layers=model_params['layers'],
                               drops=model_params['ps'],
                               input_p=model_params['dps'][0],
                               weight_p=model_params['dps'][1],
                               embed_p=model_params['dps'][2],
                               hidden_p=model_params['dps'][3],
                               qrnn=model_params['qrnn'])

    # Load the trained classifier
    model.load_state_dict(model_state)
    
    # Put the classifier into evaluation mode
    model.reset()
    model.eval()

    return stoi, classes, model

def predict_text(stoi, model, lang, text):
    """Do the actual prediction on the text using the model and mapping files passed"""

    # Predictions are done on arrays of input.
    # We only have a single input, so turn it into a 1x1 array
    texts = [text]

    # Tokenize using the fastai wrapper around spaCy
    pre_rules = [custom_tagstrip] + defaults.text_pre_rules
    tokens = Tokenizer(lang=lang, pre_rules=pre_rules, n_cpus=1).process_all(texts)

    # Turn into integers for each word
    encoded = np.array([[stoi[o] for o in p] for p in tokens], dtype=np.int64)
    
    # Turn this array into a tensor
    data = torch.from_numpy(encoded)

    # Do the predictions
    predictions = model(data)
    
    # Get class probability from classifier predictions
    res = F.softmax(predictions[0], -1).detach().cpu().numpy()
    
    return res[0]

def init():
    global stoi
    global classes
    global model
    
    # Retrieve the path to the model file using the model name
    model_path = Model.get_model_path(model_name='ps-fastai-nlp-classification')
    stoi, classes, model = load_model(model_path)

def run(raw_data):
    deser_obj = json.loads(raw_data)
    
    if not set(deser_obj.keys()) == {'lang', 'text' }:
        return { "error": "invalid data" }
    
    lang = deser_obj['lang']
    text = deser_obj['text']
    
    # Make prediction  
    scores = predict_text(stoi, model, lang, text)
    pred_class = np.argmax(scores)
    
    # You can return any data type as long as it is JSON-serializable
    # We have to cast numpy data types (non-serializable) to standard types
    return { "label": classes[pred_class], "label_index": int(pred_class), "label_score": float(scores[pred_class]), "all_scores": scores.tolist() }
