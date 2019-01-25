from flask import Flask
from flask import request
from werkzeug.http import parse_options_header
import score_local

app = Flask(__name__)

score_local.init()

@app.route('/', methods=['GET'])
def health_probe():
    return "Flask is running"
    
@app.route('/score', methods=['GET'])
def get_score():   
    input_args = {}
    for k in request.args.keys():
        values = request.args.getlist(k)
        if len(values) == 1:
            input_args[k] = json.loads(values[0]) if is_json(values[0]) else values[0]
        else:
            value_list = []
            for v in values:
                value_list.append(json.loads(v) if is_json(v) else v)

            input_args[k] = value_list

    service_input = json.dumps(input_args) 

    # run the user-provided run function
    return score_local.run(service_input)


@app.route('/score', methods=['POST'])
def post_score():

    # enforce content-type json as either the sdk or the user code is expected to json deserialize this
    if 'Content-Type' not in request.headers or parse_options_header(request.headers['Content-Type'])[0] != 'application/json':
        return json.dumps({"message": "Expects Content-Type to be application/json"})

    # expect the response to be utf-8 encodeable
    service_input = request.data.decode("utf-8")

    # run the user-provided run function
    return score_local.run(service_input)