import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .apps import PhishingurldetectionappConfig
from .feature import featureExtraction
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class Home(APIView):
    def get(self, request):
        response_dict = {"message": "Welcome! Use /api/?url=https://example.com to get predictions."}
        return Response(response_dict, status=200)

class Prediction(APIView):
    def get(self, request):
        url = request.GET.get('url')

        if not url:
            return Response({"error": "URL parameter is missing"}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure URL has a scheme
        if not urlparse(url).scheme:
            url = 'http://' + url

        logger.debug(f"URL received: {url}")

        feature_names = [
            'having_ip_address', 'long_url', 'shortening_service', 'having_@_symbol',
            'redirection_//_symbol', 'prefix_suffix_seperation', 'sub_domains', 'https_token',
            'age_of_domain', 'dns_record', 'web_traffic', 'domain_registration_length',
            'statistical_report', 'iframe', 'mouse_over'
        ]

        try:
            logger.debug("Extracting features...")
            features = np.array(featureExtraction(url)).reshape(1, -1)
            test_data = pd.DataFrame(features, columns=feature_names)

            model = PhishingurldetectionappConfig.model
            prediction = model.predict(test_data)[0]
            proba = model.predict_proba(test_data)[0]

            response = {
                "url": url,
                "featureExtractionResult": features.tolist(),
                "prediction": int(prediction),
                "successRate": round(proba[0] * 100, 2),
                "phishRate": round(proba[1] * 100, 2),
            }

            return Response(response, status=200)

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return Response({"error": f"Internal server error: {str(e)}"}, status=500)
