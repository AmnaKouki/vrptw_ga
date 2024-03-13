from util import *
import fnmatch

#transformer tous les fichiers TXT en fichiers JSON
def textTojson():
    text_data_path = os.path.join(BASE_DIR, 'data')
    json_data_path = os.path.join(BASE_DIR, 'data', 'json')
    
    for directory in [text_data_path, json_data_path]:
     if not os.path.exists(directory):
        os.makedirs(directory)

    print(text_data_path)
    for text_file in map(lambda text_filename: os.path.join(text_data_path, text_filename), \
        fnmatch.filter(os.listdir(text_data_path), '*.txt')):
        json_data = {}
        
        #'rt' : read Tesxt mode
        with io.open(text_file, 'rt', encoding='utf-8', newline='') as file_object:
            for line_count, line in enumerate(file_object, start=1):
                if line_count == 1:
                # Nom du fichier txt
                    json_data['instance_name'] = line.strip()
                    
                    
                elif line_count == 5:
                # identifier Maximum vehicle number <Vehicle capacity>    
                    values = line.strip().split()
                    json_data['max_vehicle_number'] = int(values[0])
                    json_data['vehicle_capacity'] = float(values[1])
                
                elif line_count in [2, 3, 4, 6, 7, 8, 9]:
                # ignorer ces lignes
                    pass

                elif line_count == 10:
                # ligne du Dépot
                    values = line.strip().split()
                    json_data['customer_0'] = {
                        'coordinates': {
                            'x': float(values[1]),
                            'y': float(values[2]),
                        },
                        'demand': float(values[3]),
                        'ready_time': float(values[4]),
                        'due_time': float(values[5]),
                        'service_time': float(values[6]),
                    }
                else:
                # 11 =< count =<110
                #la ligne représente un client
                    values = line.strip().split()
                    json_data[f'customer_{values[0]}'] = {
                        'coordinates': {
                            'x': float(values[1]),
                            'y': float(values[2]),
                        },
                        'demand': float(values[3]),
                        'ready_time': float(values[4]),
                        'due_time': float(values[5]),
                        'service_time': float(values[6]),
                    }
                    
        # creation de la matrice de distance (100,100)
        customers = ['customer_0'] + [f'customer_{x}' for x in range(1, 101)]
        json_data['distance_matrix'] = [[calculate_distance1(json_data[customer1], \
            json_data[customer2]) for customer1 in customers] for customer2 in customers]
        

        #sauvegarde du fichier json dans le dossier json
        json_file_name = f"{json_data['instance_name']}.json"
        json_file = os.path.join(json_data_path, json_file_name)
        print(f'Saving file: {json_file}')
        #creation du dossier des json
        make_directory_for_file(path_name=json_file)
        # 'wt' : write Text mode
        with io.open(json_file, 'wt', encoding='utf-8', newline='') as file_object:
            dump(json_data, file_object, sort_keys=True, indent=4, separators=(',', ': '))
 

textTojson()
