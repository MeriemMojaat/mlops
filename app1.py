from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests

# Initialize or create Flask instance
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "State": request.form["State"],
        "Account_length": int(request.form["Account_length"]),
        "Area_code": int(request.form["Area_code"]),
        "International_plan": request.form["International_plan"],
        "Voice_mail_plan": request.form["Voice_mail_plan"],
        "Number_vmail_messages": int(request.form["Number_vmail_messages"]),
        "Total_day_minutes": float(request.form["Total_day_minutes"]),
        "Total_day_calls": int(request.form["Total_day_calls"]),
        "Total_day_charge": float(request.form["Total_day_charge"]),
        "Total_eve_minutes": float(request.form["Total_eve_minutes"]),
        "Total_eve_calls": int(request.form["Total_eve_calls"]),
        "Total_eve_charge": float(request.form["Total_eve_charge"]),
        "Total_night_minutes": float(request.form["Total_night_minutes"]),
        "Total_night_calls": int(request.form["Total_night_calls"]),
        "Total_night_charge": float(request.form["Total_night_charge"]),
        "Total_intl_minutes": float(request.form["Total_intl_minutes"]),
        "Total_intl_calls": int(request.form["Total_intl_calls"]),
        "Total_intl_charge": float(request.form["Total_intl_charge"]),
        "Customer_service_calls": int(request.form["Customer_service_calls"]),
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict", json=input_data, timeout=5
        )
        response.raise_for_status()
        prediction = response.json().get("prediction", "No prediction received.")
    except requests.exceptions.RequestException as e:
        return jsonify(error=str(e), prediction=None), 500

    # Redirect to the prediction result page
    return redirect(url_for("result", prediction=prediction))


@app.route("/result")
def result():
    prediction = request.args.get("prediction", "No prediction found")
    return render_template("result.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=False)
