#!/usr/bin/env python3

import numpy as np
import sys, os

# Event types
ARRIVAL = 'arrival'
DEPARTURE = 'departure'

class ServerFarm:
    def __init__(self, n_servers, threshold):
        self.n_servers = n_servers              # Total number of servers
        self.threshold = threshold              # Sub-job threshold to distinguish between high and low priority queues
        self.high_queue, self.low_queue = [], []  # Separate queues for jobs
        self.servers = [None] * n_servers       # Tracks current job on each server
        self.master_clock = 0                   # Global simulation clock
        self.subjob_records = []                # Stores sub-job (arrival_time, departure_time, sort_key)
        self.job_completion = dict()            # Records completion times for each job
        self.task_id = 0                        # Job ID counter

    def process_arrival(self, arrival_time, sub_jobs):
        self.master_clock = arrival_time
        for sub_idx, service_time in enumerate(sub_jobs):
            subjob_id = (self.task_id, sub_idx)
            target_queue = self.high_queue if len(sub_jobs) <= self.threshold else self.low_queue
            target_queue.append((arrival_time, service_time, subjob_id))
        self.task_id += 1
        self.dispatch_jobs()

    def process_departure(self, server_idx):
        start_time, departure, subjob_id = self.servers[server_idx]
        self.master_clock = departure
        job_id = subjob_id[0]
        arrival_for_output = self.arrival_map[job_id]
        self.subjob_records.append((arrival_for_output, departure, departure))
        self.servers[server_idx] = None

        if job_id not in self.job_completion:
            self.job_completion[job_id] = []
        self.job_completion[job_id].append(departure)

        self.dispatch_jobs()

    def dispatch_jobs(self):
        for idx, server in enumerate(self.servers):
            if server is None:  # If server is free
                queue = self.high_queue if self.high_queue else self.low_queue
                if queue:
                    arrival, service, subjob_id = queue.pop(0)
                    start_time = self.master_clock
                    departure = start_time + service
                    self.servers[idx] = (start_time, departure, subjob_id)

    def next_event(self, next_arrival_time):
        # Determine which is earlier: next arrival or next departure
        active_departures = [(idx, server[1]) for idx, server in enumerate(self.servers) if server is not None]

        if active_departures:
            server_idx, next_departure_time = min(active_departures, key=lambda x: x[1])
        else:
            server_idx = None
            next_departure_time = np.inf

        if next_arrival_time < next_departure_time:
            return ARRIVAL, next_arrival_time
        else:
            return DEPARTURE, server_idx

    def simulate(self, arrivals, services, time_end=np.inf):
        self.arrival_map = {i: arrivals[i] for i in range(len(arrivals))}

        events = []
        arrivals_iter = iter(zip(arrivals, services))
        try:
            next_arrival, next_service = next(arrivals_iter)
        except StopIteration:
            next_arrival = np.inf

        while self.master_clock < time_end:
            event_type, event_info = self.next_event(next_arrival)

            if event_type == ARRIVAL:
                self.process_arrival(next_arrival, next_service)
                events.append((next_arrival, ARRIVAL))
                try:
                    next_arrival, next_service = next(arrivals_iter)
                except StopIteration:
                    next_arrival = np.inf

            elif event_type == DEPARTURE and event_info is not None:
                self.process_departure(event_info)
                events.append((self.master_clock, DEPARTURE))

            # Stop if no more events
            if next_arrival == np.inf and all(s is None for s in self.servers):
                break

        return events

def generate_random_trace(time_end, lambda_rate, a2l, a2u, subjob_probs, mu, alpha, seed=None):
    # Generate random arrivals and service times
    arrivals = []
    services = []
    time = 0
    rng = np.random.default_rng(seed)

    while time < time_end:
        interarrival = rng.exponential(1 / lambda_rate) * rng.uniform(a2l, a2u)
        time += interarrival
        if time >= time_end:
            break

        arrivals.append(time)
        num_subjobs = rng.choice(np.arange(1, len(subjob_probs)+1), p=subjob_probs)
        sub_services = rng.weibull(alpha, num_subjobs) / mu
        services.append(sub_services)

    return np.array(arrivals), services

def read_trace_files(config_id):
    try:
        arrivals = np.cumsum(np.loadtxt(f'config/interarrival_{config_id}.txt'))
        service_times = np.loadtxt(f'config/service_{config_id}.txt')
        services = [row[~np.isnan(row)] for row in service_times]
        return arrivals, services
    except OSError:
        return None, None

def main(config_id):
    # Read mode (trace/random)
    with open(f'config/mode_{config_id}.txt', 'r') as f:
        mode = f.read().strip()

    # Read parameters
    with open(f'config/para_{config_id}.txt', 'r') as f:
        lines = f.read().splitlines()
        n, h = int(lines[0]), int(lines[1])
        time_end = float(lines[2]) if mode == 'random' and len(lines) > 2 else np.inf

    if mode == 'trace':
        arrivals, services = read_trace_files(config_id)
    else:
        interarrival_params = np.loadtxt(f'config/interarrival_{config_id}.txt', max_rows=1)
        subjob_probs = np.loadtxt(f'config/interarrival_{config_id}.txt', skiprows=1)
        lambda_rate, a2l, a2u = interarrival_params
        mu, alpha = np.loadtxt(f'config/service_{config_id}.txt')
        arrivals, services = generate_random_trace(time_end, lambda_rate, a2l, a2u, subjob_probs, mu, alpha)

    # Initialize and run simulation
    sf = ServerFarm(n_servers=n, threshold=h)
    events = sf.simulate(arrivals, services, time_end)

    os.makedirs('output', exist_ok=True)

    # Write departure info
    with open(f'output/dep_{config_id}.txt', 'w') as dep_file:
        for arrival, departure, sort_key in sorted(sf.subjob_records, key=lambda x: x[2]):
            dep_file.write(f'{arrival:.4f}\t{departure:.4f}\n')

    # Calculate mean response time (MRT)
    job_response_times = []
    for job_id, departure_times in sf.job_completion.items():
        response = max(departure_times) - arrivals[job_id]
        job_response_times.append(response)

    mrt = np.mean(job_response_times)
    with open(f'output/mrt_{config_id}.txt', 'w') as mrt_file:
        mrt_file.write(f'{mrt:.4f}\n')

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python main.py <config_id>')
    else:
        main(sys.argv[1])
