import json
import time
import math
import heapq
import os
import tracemalloc
from typing import List, Dict

# Colorfull
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

class City:
    """Representa uma cidade com seus atributos geográficos e demográficos."""
    def __init__(self, data):
        self.codigo = data["codigo"]
        self.municipio = data["municipio"]
        self.latitude = data["latitude"]
        self.longitude = data["longitude"]
        self.populacao = data.get("populacao", float('inf'))
        self.distancia_maceio = data.get("distanciaMaceio", "")
        self.tempo_viagem_maceio = data.get("tempoViagemMaceio", "")
        self.display_id = ""
    
    def __str__(self):
        return f"{self.municipio} ({self.display_id})"
    
    def __repr__(self):
        return self.__str__()

#=================================================================
# LOAD DF E CÁLCULO
#=================================================================

def load_cities(file_path: str) -> List[City]:
    """Load objects"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            cities = []
            for i, city_data in enumerate(data):
                if "latitude" in city_data and "longitude" in city_data:
                    city = City(city_data)
                    city.display_id = f"{i:02d}"
                    cities.append(city)
            return cities
    except FileNotFoundError:
        print(f"{Colors.RED}Arquivo não encontrado: {file_path}{Colors.END}")
        return []
    except json.JSONDecodeError:
        print(f"{Colors.RED}Erro ao decodificar o arquivo JSON: {file_path}{Colors.END}")
        return []

def calculate_euclidean_distance(city1: City, city2: City) -> float:
    """Calcula distância euclidiana dos dois pontos"""
    lat_diff = city1.latitude - city2.latitude
    lon_diff = city1.longitude - city2.longitude
    return math.sqrt(lat_diff**2 + lon_diff**2)

def find_neighbors(city: City, all_cities: List[City], radius: float) -> List[City]:
    """Encontra as cidades vizinhas dentro do raio"""
    neighbors = []
    for potential_neighbor in all_cities:
        if city.codigo == potential_neighbor.codigo:
            continue
        distance = calculate_euclidean_distance(city, potential_neighbor)
        if distance <= radius:
            neighbors.append(potential_neighbor)
    return neighbors

def estimate_travel_time(path: List[City]) -> float:
    """Calculo pra estimar o tempo total de viagem (min)"""
    total_minutes = 0
    for i in range(len(path) - 1):
        city1 = path[i]
        city2 = path[i + 1]
        distance = calculate_euclidean_distance(city1, city2)
        time_str = city2.tempo_viagem_maceio
        
        if "hora" in time_str and "min" in time_str:
            parts = time_str.split("hora")
            hours = int(parts[0].strip())
            mins_part = parts[1].split("min")[0].strip()
            mins = int(mins_part)
            total_time = hours * 60 + mins
        elif "min" in time_str:
            mins = int(time_str.split("min")[0].strip())
            total_time = mins
        else:
            estimated_km = distance * 111
            total_time = (estimated_km / 60) * 60
            
        try:
            dist_maceio = float(city2.distancia_maceio.split(" ")[0])
            factor = distance / (dist_maceio / 111)
            adjusted_time = total_time * factor
            total_minutes += adjusted_time
        except:
            estimated_km = distance * 111
            total_minutes += (estimated_km / 60) * 60
            
    return total_minutes

#=================================================================
# ALGORITMOS APLICADOS
#=================================================================

def dijkstra_search(start_city: City, end_city: City, all_cities: List[City], 
                   radius: float, penalty: float = 0.0) -> Dict:
    """Algoritmo de busca de Dijkstra"""
    tracemalloc.start()
    start_time = time.time()
    distances = {city.codigo: float('inf') for city in all_cities}
    distances[start_city.codigo] = 0
    previous = {city.codigo: None for city in all_cities}
    visited = set()
    nodes_visited = 0
    pq = [(0, 0, start_city.populacao, start_city)]
    counter = 1
    
    while pq:
        current_distance, _, _, current_city = heapq.heappop(pq)
        if current_city.codigo in visited:
            continue
        visited.add(current_city.codigo)
        nodes_visited += 1
        
        if current_city.codigo == end_city.codigo:
            break
            
        neighbors = find_neighbors(current_city, all_cities, radius)
        for neighbor in neighbors:
            if neighbor.codigo in visited:
                continue
            distance = calculate_euclidean_distance(current_city, neighbor)
            penal = penalty * neighbor.populacao
            new_distance = current_distance + distance + penal
            
            if new_distance < distances[neighbor.codigo]:
                distances[neighbor.codigo] = new_distance
                previous[neighbor.codigo] = current_city
                heapq.heappush(pq, (new_distance, counter, neighbor.populacao, neighbor))
                counter += 1
    
    path = []
    current = end_city
    if previous[end_city.codigo] is not None or start_city.codigo == end_city.codigo:
        while current:
            path.append(current)
            if current.codigo == start_city.codigo:
                break
            current = previous[current.codigo]
        path.reverse()
        
    execution_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()  # Obtém o uso de memória
    tracemalloc.stop()
    
    return {
        "success": len(path) > 0,
        "path": path,
        "distance": distances[end_city.codigo],
        "execution_time": execution_time,
        "nodes_visited": nodes_visited,
        "memory_usage": peak_memory / 1024
    }

def astar_search(start_city: City, end_city: City, all_cities: List[City], 
                radius: float, penalty: float = 0.0) -> Dict:
    """Algoritmo A*"""
    tracemalloc.start() 
    start_time = time.time()
    
    def heuristic(city: City) -> float:
        return calculate_euclidean_distance(city, end_city) + penalty * city.populacao
    
    g_score = {city.codigo: float('inf') for city in all_cities}
    g_score[start_city.codigo] = 0
    f_score = {city.codigo: float('inf') for city in all_cities}
    f_score[start_city.codigo] = heuristic(start_city)
    
    open_set = [(f_score[start_city.codigo], 0, start_city.populacao, start_city)]
    came_from = {city.codigo: None for city in all_cities}
    closed_set = set()
    counter = 1
    nodes_visited = 0
    
    while open_set:
        _, _, _, current = heapq.heappop(open_set)
        if current.codigo == end_city.codigo:
            break
        if current.codigo in closed_set:
            continue
            
        closed_set.add(current.codigo)
        nodes_visited += 1
        neighbors = find_neighbors(current, all_cities, radius)
        
        for neighbor in neighbors:
            if neighbor.codigo in closed_set:
                continue
                
            penal = penalty * neighbor.populacao
            tentative_g_score = g_score[current.codigo] + calculate_euclidean_distance(current, neighbor) + penal
            
            if tentative_g_score < g_score[neighbor.codigo]:
                came_from[neighbor.codigo] = current
                g_score[neighbor.codigo] = tentative_g_score
                f_score[neighbor.codigo] = tentative_g_score + heuristic(neighbor)
                heapq.heappush(open_set, (f_score[neighbor.codigo], counter, neighbor.populacao, neighbor))
                counter += 1
    
    path = []
    current = end_city
    if came_from[end_city.codigo] is not None or start_city.codigo == end_city.codigo:
        while current:
            path.append(current)
            if current.codigo == start_city.codigo:
                break
            current = came_from[current.codigo]
        path.reverse()
        
    execution_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()  # Obtém o uso de memória
    tracemalloc.stop()
    
    return {
        "success": len(path) > 0,
        "path": path,
        "distance": g_score[end_city.codigo],
        "execution_time": execution_time,
        "nodes_visited": nodes_visited,
        "memory_usage": peak_memory / 1024
    }

def bidirectional_dijkstra(start_city: City, end_city: City, all_cities: List[City], 
                          radius: float, penalty: float = 0.0) -> Dict:
    """Algoritmo Dijkstra bidirecional"""
    tracemalloc.start()
    start_time = time.time()
    
    distances_f = {city.codigo: float('inf') for city in all_cities}
    distances_b = {city.codigo: float('inf') for city in all_cities}
    prev_f = {city.codigo: None for city in all_cities}
    prev_b = {city.codigo: None for city in all_cities}
    distances_f[start_city.codigo] = 0
    distances_b[end_city.codigo] = 0
    
    pq_f = [(0, 0, start_city.populacao, start_city)]
    pq_b = [(0, 0, end_city.populacao, end_city)]
    visited_f, visited_b = {}, {}
    counter_f = counter_b = 1
    meet_node = None
    nodes_visited = 0
    
    while pq_f and pq_b:
        if pq_f:
            dist_f, _, _, current_f = heapq.heappop(pq_f)
            if current_f.codigo in visited_f:
                continue
            visited_f[current_f.codigo] = dist_f
            nodes_visited += 1
            
            if current_f.codigo in visited_b:
                meet_node = current_f.codigo
                break
                
            neighbors = find_neighbors(current_f, all_cities, radius)
            for neighbor in neighbors:
                penal = penalty * neighbor.populacao
                new_dist = dist_f + calculate_euclidean_distance(current_f, neighbor) + penal
                
                if new_dist < distances_f[neighbor.codigo]:
                    distances_f[neighbor.codigo] = new_dist
                    prev_f[neighbor.codigo] = current_f
                    heapq.heappush(pq_f, (new_dist, counter_f, neighbor.populacao, neighbor))
                    counter_f += 1
        
        if pq_b:
            dist_b, _, _, current_b = heapq.heappop(pq_b)
            if current_b.codigo in visited_b:
                continue
            visited_b[current_b.codigo] = dist_b
            nodes_visited += 1
            
            if current_b.codigo in visited_f:
                meet_node = current_b.codigo
                break
                
            neighbors = find_neighbors(current_b, all_cities, radius)
            for neighbor in neighbors:
                penal = penalty * neighbor.populacao
                new_dist = dist_b + calculate_euclidean_distance(current_b, neighbor) + penal
                
                if new_dist < distances_b[neighbor.codigo]:
                    distances_b[neighbor.codigo] = new_dist
                    prev_b[neighbor.codigo] = current_b
                    heapq.heappush(pq_b, (new_dist, counter_b, neighbor.populacao, neighbor))
                    counter_b += 1
    
    if not meet_node:
        return {
            "success": False, 
            "path": [], 
            "distance": float('inf'),
            "execution_time": time.time() - start_time, 
            "nodes_visited": nodes_visited,
            "memory_usage": 0
        }
    
    path_f = []
    current_code = meet_node
    while current_code is not None:
        city = next((c for c in all_cities if c.codigo == current_code), None)
        path_f.append(city)
        current_code = prev_f[current_code].codigo if prev_f[current_code] else None
    path_f.reverse()

    path_b = []
    current_code = prev_b[meet_node].codigo if prev_b[meet_node] else None
    while current_code is not None:
        city = next((c for c in all_cities if c.codigo == current_code), None)
        path_b.append(city)
        current_code = prev_b[current_code].codigo if prev_b[current_code] else None

    path = path_f + path_b
    execution_time = time.time() - start_time
    current_memory, peak_memory = tracemalloc.get_traced_memory()  # Obtém o uso de memória
    tracemalloc.stop()
    
    return {
        "success": True,
        "path": path,
        "distance": distances_f[meet_node] + distances_b[meet_node],
        "execution_time": execution_time,
        "nodes_visited": nodes_visited,
        "memory_usage": peak_memory / 1024
    }

#=================================================================
# UPGRADE VIEW
#=================================================================

def display_cities_table(cities: List[City], columns: int = 4) -> None:
    """Exibe a lista de cidades 4 columns"""
    print(f"\n{Colors.BOLD}===== LISTA DE CIDADES ====={Colors.END}")

    columns = 4

    max_name_length = max(len(city.municipio) for city in cities)
    column_width = max_name_length + 4
    
    header = f"{Colors.BLUE}{'ID. Município':<{column_width}}{Colors.END}"
    print(header * columns)
    print("-" * (column_width * columns))
    
    for i in range(0, len(cities), columns):
        row = ""
        for j in range(columns):
            if i + j < len(cities):
                city = cities[i + j]
                entry = f"{city.display_id}. {city.municipio}"
                row += f"{entry:<{column_width}}"
            else:
                row += " " * column_width
        print(row)
    
    print(f"{Colors.BOLD}{'-' * (column_width * columns)}{Colors.END}")

def display_city_connections(city: City, all_cities: List[City], radius: float) -> None:
    """Exibe as conexões da cidade dentro do raio"""
    neighbors = find_neighbors(city, all_cities, radius)
    
    if not neighbors:
        print(f"\n{Colors.YELLOW}{city.municipio} não tem conexões diretas com o raio {radius}{Colors.END}")
        return
    
    print(f"\n{Colors.BOLD}Conexões de {Colors.GREEN}{city.municipio}{Colors.END}:")
    print(f"  │")
    
    neighbors_with_dist = [(n, calculate_euclidean_distance(city, n)) for n in neighbors]
    neighbors_with_dist.sort(key=lambda x: x[1])
    
    for i, (neighbor, distance) in enumerate(neighbors_with_dist):
        km_distance = distance * 111
        is_last = i == len(neighbors_with_dist) - 1
        branch = "└─" if is_last else "├─"
        print(f"  {branch} {neighbor.municipio} ({distance:.4f} un, ~{km_distance:.1f} km)")

def display_graph_representation(cities: List[City], radius: float) -> None:
    """Estatísticas e informações do grafo de cidades."""
    print(f"\n{Colors.BOLD}===== REPRESENTAÇÃO DO GRAFO ====={Colors.END}")
    print(f"{Colors.YELLOW}Raio de conexão: {radius} unidades (~{radius*111:.1f} km){Colors.END}")
    
    total_connections = 0
    city_connections = {}
    for city in cities:
        neighbors = find_neighbors(city, cities, radius)
        city_connections[city.municipio] = len(neighbors)
        total_connections += len(neighbors)
    
    most_connected = max(city_connections.items(), key=lambda x: x[1])
    least_connected = min(city_connections.items(), key=lambda x: x[1])
    
    print(f"• Total de cidades: {Colors.BOLD}{len(cities)}{Colors.END}")
    print(f"• Total de conexões: {Colors.BOLD}{total_connections//2}{Colors.END} (cada conexão contada uma vez)")
    print(f"• Cidade mais conectada: {Colors.GREEN}{most_connected[0]}{Colors.END} com {Colors.BOLD}{most_connected[1]}{Colors.END} conexões")
    print(f"• Cidade menos conectada: {Colors.RED}{least_connected[0]}{Colors.END} com {Colors.BOLD}{least_connected[1]}{Colors.END} conexões")

def display_route_visualization(path: List[City]) -> None:
    """Visualização da rota encontrada."""
    if not path or len(path) < 2:
        print(f"\n{Colors.YELLOW}Não há rota para visualizar.{Colors.END}")
        return
    
    print(f"\n{Colors.BOLD}===== VISUALIZAÇÃO DA ROTA ====={Colors.END}")
    total_distance = 0
    
    print(f"{Colors.GREEN}┌──[INÍCIO] {path[0].municipio}{Colors.END}")
    
    for i in range(1, len(path)):
        prev_city = path[i-1]
        current_city = path[i]
        distance = calculate_euclidean_distance(prev_city, current_city)
        total_distance += distance
        
        print(f"{Colors.BLUE}│   {Colors.END}")
        print(f"{Colors.BLUE}│   {distance:.4f} un (~{distance*111:.1f} km){Colors.END}")
        
        if i == len(path) - 1:
            print(f"{Colors.RED}└──[FIM] {current_city.municipio}{Colors.END}")
        else:
            print(f"{Colors.BLUE}├──{Colors.END} {current_city.municipio}")
    
    print(f"\n{Colors.BOLD}Distância total: {Colors.YELLOW}{total_distance:.4f} un (~{total_distance*111:.1f} km){Colors.END}")
    print(f"{Colors.BOLD}==============================={Colors.END}")

def format_results(result: Dict, algorithm_name: str) -> None:
    """Formata e exibe os resultados de um algoritmo de busca."""
    border = "═" * 60
    print(f"\n{Colors.BOLD}{border}{Colors.END}")
    print(f"{Colors.BOLD}{algorithm_name:^60}{Colors.END}")
    print(f"{Colors.BOLD}{border}{Colors.END}")
    
    if not result["success"]:
        print(f"{Colors.RED}Não foi possível encontrar uma rota entre as cidades com o raio especificado.{Colors.END}")
        print(f"{Colors.BOLD}{border}{Colors.END}")
        return
    
    path = [city.municipio for city in result["path"]]
    print(f"Rota: {Colors.GREEN}{' → '.join(path)}{Colors.END}")
    print(f"Distância total: {Colors.YELLOW}{result['distance']:.4f} graus{Colors.END} | {Colors.YELLOW}{result['distance']*111:.2f} km{Colors.END}")
    print(f"Número de cidades no caminho: {Colors.BOLD}{len(path)}{Colors.END}")
    print(f"Nós visitados: {Colors.BOLD}{result['nodes_visited']}{Colors.END}")
    print(f"Tempo de execução: {Colors.BOLD}{result['execution_time']:.6f} segundos{Colors.END}")
    print(f"Uso de memória (pico): {Colors.YELLOW}{result['memory_usage']:.2f} KB{Colors.END}")

    if len(result["path"]) > 1:
        try:
            total_time_mins = estimate_travel_time(result["path"])
            hours = int(total_time_mins // 60)
            mins = int(total_time_mins % 60)
            print(f"Tempo estimado de viagem: {Colors.BLUE}{hours}h {mins}min{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}Não foi possível estimar o tempo de viagem: {str(e)}{Colors.END}")
    
    print(f"{Colors.BOLD}{border}{Colors.END}")

def print_comparison(dijkstra_result: Dict, astar_result: Dict, bidi_result: Dict) -> None:
    """Exibe uma comparação tabular entre os resultados dos três algoritmos."""
    border = "═" * 70
    print(f"\n{Colors.BOLD}{border}{Colors.END}")
    print(f"{Colors.BOLD}{'COMPARAÇÃO ENTRE OS ALGORITMOS':^70}{Colors.END}")
    print(f"{Colors.BOLD}{border}{Colors.END}")
    
    print(f"{Colors.BLUE}{'Algoritmo':<20} {'Cidades':<10} {'Distância (km)':<15} {'Nós':<8} {'Tempo (s)':<10}{Colors.END}")
    print("─" * 70)
    
    print(f"{'Dijkstra':<20} {len(dijkstra_result['path']):<10} {dijkstra_result['distance']*111:<15.2f} {dijkstra_result['nodes_visited']:<8} {dijkstra_result['execution_time']:<10.6f}")
    print(f"{'A*':<20} {len(astar_result['path']):<10} {astar_result['distance']*111:<15.2f} {astar_result['nodes_visited']:<8} {astar_result['execution_time']:<10.6f}")
    print(f"{'Bidirecional':<20} {len(bidi_result['path']):<10} {bidi_result['distance']*111:<15.2f} {bidi_result['nodes_visited']:<8} {bidi_result['execution_time']:<10.6f}")
    print(f"{Colors.BOLD}{border}{Colors.END}")
    
    if not (dijkstra_result["distance"] == astar_result["distance"] == bidi_result["distance"]):
        print(f"{Colors.YELLOW}Atenção: As distâncias totais diferem entre os algoritmos!{Colors.END}")
    else:
        print(f"{Colors.GREEN}Todos os algoritmos encontraram a mesma distância total.{Colors.END}")
    
    fastest = min([
        ("Dijkstra", dijkstra_result["execution_time"]), 
        ("A*", astar_result["execution_time"]), 
        ("Bidirecional", bidi_result["execution_time"])
    ], key=lambda x: x[1])
    
    fewest_nodes = min([
        ("Dijkstra", dijkstra_result["nodes_visited"]), 
        ("A*", astar_result["nodes_visited"]), 
        ("Bidirecional", bidi_result["nodes_visited"])
    ], key=lambda x: x[1])
    
    print(f"• Algoritmo mais rápido: {Colors.GREEN}{fastest[0]}{Colors.END} ({fastest[1]:.6f}s)")
    print(f"• Algoritmo mais eficiente (menos nós): {Colors.GREEN}{fewest_nodes[0]}{Colors.END} ({fewest_nodes[1]} nós)")
    
    print(f"{Colors.BOLD}{border}{Colors.END}")

#=================================================================
# MAIN
#=================================================================

def main():
    # start clean terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'BUSCA DE ROTAS ENTRE CIDADES DE ALAGOAS':^60}{Colors.END}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.END}")
    
    file_path = "cities-al.json"
    cities = load_cities(file_path)
    
    display_cities_table(cities)
    
    try:
        start_id = input(f"{Colors.BOLD}Digite o código da cidade de origem: {Colors.END}").strip()
        start_city = next((city for city in cities if city.display_id == start_id), None)
            
        end_id = input(f"{Colors.BOLD}Digite o código da cidade de destino: {Colors.END}").strip()
        end_city = next((city for city in cities if city.display_id == end_id), None)
        
        # Parameters
        radius = 0.22  # Raio de conexão (graus)
        penalty = 0.0000005  # Penalidade populacional
        
        print(f"\n{Colors.YELLOW}Usando raio de conexão: {radius} (aprox. {radius*111:.1f} km){Colors.END}")
        
        display_graph_representation(cities, radius)
        display_city_connections(start_city, cities, radius)
        
        print(f"\n{Colors.BOLD}Buscando rota de {Colors.GREEN}{start_city.municipio}{Colors.END} para {Colors.RED}{end_city.municipio}{Colors.END} com raio {radius}...{Colors.END}")
        
        # Call methods
        # Dijkstra
        dijkstra_result = dijkstra_search(start_city, end_city, cities, radius, penalty)
        format_results(dijkstra_result, "Dijkstra (penalidade população)")
        if dijkstra_result["success"]:
            display_route_visualization(dijkstra_result["path"])
        
        # A*
        astar_result = astar_search(start_city, end_city, cities, radius, penalty)
        format_results(astar_result, "A* (penalidade população)")
        if astar_result["success"] and str(dijkstra_result["path"]) != str(astar_result["path"]):
            display_route_visualization(astar_result["path"])
        
        # Dijkstra Bidirecional
        bidi_result = bidirectional_dijkstra(start_city, end_city, cities, radius, penalty)
        format_results(bidi_result, "Dijkstra Bidirecional (penalidade população)")
        if bidi_result["success"]:
            display_route_visualization(bidi_result["path"])
        
        if dijkstra_result["success"] and astar_result["success"] and bidi_result["success"]:
            print_comparison(dijkstra_result, astar_result, bidi_result)
            
    except ValueError:
        print(f"{Colors.RED}Entrada inválida. Certifique-se de inserir valores numéricos onde necessário.{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Ocorreu um erro: {str(e)}{Colors.END}")

if __name__ == "__main__":
    main()