from flask import Flask
from flask_restx import Api, Resource

REST_API_PORT = 5000

app = Flask(__name__)
api = Api(app, 
          version='0.1', 
          title='Gender Inference API', 
          description='API that demos a simple inference mechanism by returning the gender of a name')

ns = api.namespace('gender')
@ns.route('/<string:firstname>')
@ns.response(200, 'Inference was successful')
@ns.response(400, 'Firstname must contain only characters')
@ns.param('firstname', 'The firstname whose gender should be returned')
class GenderInference(Resource):
    def get(self,firstname):
        """
            Inferes the gender based on a given firstname
        """
        if not firstname or not firstname.isalpha():
            api.abort(400, "Name should only contain characters")

        data = { "gender": "unknown" }

        # here we perform the "prediction"
        if firstname == "Stefan":
            data['gender'] = "male"
        elif firstname == "Nora":
            data['gender'] = "female"
        return data     
        
if __name__ == '__main__':
    app.run(port=REST_API_PORT)
