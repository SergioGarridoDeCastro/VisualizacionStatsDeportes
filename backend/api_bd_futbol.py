from flask import jsonify, Flask

api_name = "localhost/sport"

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/players/league/team/player', methods=['POST'])
def post_data_player(id):
    pass

@app.route('/league/', methods=['GET'])
def get_seasons(id):
    pass


@app.route('/league/season/', methods=['GET'])
def get_teams(id):
    pass

@app.route('/league/season/team/', methods=['GET'])
def get_matches():
    pass