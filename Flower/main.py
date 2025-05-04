import multiprocessing
import time
import subprocess

def run_superlink():
    """Lancer le SuperLink (serveur Flower)"""
    print("Démarrage du serveur SuperLink...")
    subprocess.run(["flower-superlink", "--insecure"], check=True)

def run_supernode(client_id):
    """Lancer le SuperNode (client Flower)"""
    print(f"Démarrage du client {client_id}...")
    subprocess.run([
        "python3", "/home/onyxia/work/Federated_Learning_Milliman/Flower/client.py", f"--client_id={client_id}"
    ], check=True)

def main():
    # Lancez le serveur SuperLink dans un processus séparé
    server_process = multiprocessing.Process(target=run_superlink)
    server_process.start()

    time.sleep(3)  # Attendre que le serveur SuperLink soit prêt

    # Lancez les clients dans des processus séparés
    client_processes = []
    num_clients = 2  # Nombre de clients à lancer, ajustez selon vos besoins
    for cid in range(num_clients):
        p = multiprocessing.Process(target=run_supernode, args=(cid,))
        p.start()
        client_processes.append(p)

    # Attendez que les processus clients se terminent
    for p in client_processes:
        p.join()

    # Terminer le serveur après la fin des clients
    server_process.terminate()

if __name__ == "__main__":
    main()
