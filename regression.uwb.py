from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes
from rest_framework import permissions
from rest_framework.response import Response
from datetime import datetime
from django.shortcuts import render
import csv
from uwb.models import UWBData  # Import your UWBData model here


@csrf_exempt
@api_view(['POST', 'GET'])
@permission_classes((permissions.AllowAny,))
def uwb_view(request):
    if request.method == 'POST':
        try:
            # Extract JSON data from the request
            json_data = request.data

            # Validate that the required fields are present in the JSON data
            required_fields = ['Connected MAC', 'Anchor MAC', 'Session', 'Distance', 'Azimuth', 'Elevation',
                               'Line of Sight', 'day', 'time', 'Onion Address']
            for field in required_fields:
                if field not in json_data or not json_data[field]:
                    return Response({"message": f"Missing required field: {field}", "status": 0, "results": {}})

            # Parse the JSON data
            connected_mac = json_data['Connected MAC']
            anchor_mac = json_data['Anchor MAC']
            session = json_data['Session']
            distance = json_data['Distance']
            azimuth = json_data['Azimuth']
            elevation = json_data['Elevation']
            line_of_sight = json_data['Line of Sight']
            day = json_data['day']
            time = json_data['time']
            onion_address = json_data['Onion Address']
            # port = json_data['Port']

            # Here, you should create and save an instance of your UWBData model with the received data
            uwb_data = UWBData(
                Connected_MAC=connected_mac,
                Anchor_MAC=anchor_mac,
                Session=session,
                Distance=distance,
                Azimuth=azimuth,
                Elevation=elevation,
                Line_of_Sight=line_of_sight,
                day=day,
                time=time,
                Onion_Address=onion_address,
                # Port=port,
            )

            uwb_data.save()
            def export_csv(request):
                try:
        # Retrieve all UWBData entries
        uwb_entries = UWBData.objects.all()
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="uwb_data_export.csv"'

        writer = csv.writer(response)
        writer.writerow(
            ['Connected_MAC', 'Anchor_MAC', 'Session', 'Distance', 'Azimuth', 'Elevation', 'Line_of_Sight', 'day',
             'time', 'Onion_Address'])

        for entry in uwb_entries:
            writer.writerow(
                [entry.Connected_MAC, entry.Anchor_MAC, entry.Session, entry.Distance, entry.Azimuth, entry.Elevation,
                 entry.Line_of_Sight, entry.day, entry.time, entry.Onion_Address])

    return response

except Exception as e:
     return HttpResponse(f"Error exporting CSV: {str(e)}", content_type='text/plain')

return Response({"message": "UWB data logged successfully", "status": 1, "results": {"Connected_MAC": connected_mac}})
except Exception as e:
return Response({"message": f"Error processing UWB data: {str(e)}", "status": 0, "results": {}})

elif request.method == 'GET':
try:
    # Query the database for UWB data (replace this with your actual query)
    uwb_data = UWBData.objects.all()  # Replace with your query
    # Serialize the data to a table format
    uwb_data_table = "<table border='1' cellpadding='5'><tr><th>Connected_MAC</th><th>Session</th><th>Distance</th><th>Azimuth</th><th>Elevation</th><th>Line_of_Sight</th><th>Day</th><th>Time</th><th>Onion_Address</th><th>Anchor_MAC</th></tr>"
    for entry in uwb_data:
        uwb_data_table += f"<tr><td>{entry.Connected_MAC}</td><td>{entry.Session}</td><td>{entry.Distance}</td><td>{entry.Azimuth}</td><td>{entry.Elevation}</td><td>{entry.Line_of_Sight}</td><td>{entry.day}</td><td>{entry.time}</td><td>{entry.Onion_Address}</td><td>{entry.Anchor_MAC}</td></tr>"
    uwb_data_table += "</table>"
    return Response({"message": "UWB data retrieved successfully", "status": 1, "results": uwb_data_table})
except Exception as e:
    return Response({"message": f"Error retrieving UWB data: {str(e)}", "status": 0, "results": []})

return Response({"message": "No data available", "status": 0, "results": []})
