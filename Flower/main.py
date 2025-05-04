import multiprocessing
import time
import subprocess

def run_superlink():
    """Lancer le SuperLink (le serveur Flower moderne)"""
    subprocess.run(["flower-superlink", "--insecure"], check=True)

def run_supernode(client_id):
    """Lancer le SuperNode (client Flower moderne)"""
    subprocess.run([
        "flower-supernode",
        "--insecure",
        "--superlink=127.0.0.1:9091",
        "--start-python", "client.py",
        "--", f"--client_id={client_id}"
    ], check=True)

if __name__ == "__main__":
    # Lancer le serveur SuperLink (port 9091 par défaut)
    server_process = multiprocessing.Process(target=run_superlink)
    server_process.start()

    time.sleep(3)  # Attendre que le SuperLink soit prêt

    # Lancer les clients
    client_processes = []
    for cid in range(2):  # Ajuste le nombre de clients ici
        p = multiprocessing.Process(target=run_supernode, args=(cid,))
        p.start()
        client_processes.append(p)

    # Attendre que les clients terminent
    for p in client_processes:
        p.join()

    # Terminer le serveur après les clients
    server_process.terminate()
