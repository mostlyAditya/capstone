from flask import Flask,jsonify,request
from flask_pymongo import PyMongo
from bson.json_util import  dumps
import pymongo
from bson.objectid import ObjectId

app = Flask(__name__)
app.secret_key = "secretkey"
#app.config['MONGO_dbname'] = 'orders'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/database'
mongo = PyMongo(app)

@app.route('/welcome', methods=['GET'])
def welcome():
    return "Welcome to Seat Hunt"

@app.route('/get_seat',methods = ['GET'])
def get_seat():
    data = mongo.db.model_data.find().sort("second")
    print(data)
    data = dumps(data[0])
    print(data)
    return data


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105, debug = True, use_reloader = True)