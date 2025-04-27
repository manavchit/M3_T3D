#include <mpi.h>         // MPI library for parallel processing
#include <iostream>      // Standard I/O operations
#include <fstream>       // File operations
#include <sstream>       // String stream operations
#include <map>           // Map container
#include <unordered_map> // Hash map container
#include <vector>        // Vector container
#include <queue>         // Queue container
#include <algorithm>     // Algorithms like sort

using namespace std;

int main(int argc, char* argv[]) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    // Get process rank and total number of processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Check command line arguments
    if (argc < 3) {
        if (rank == 0) { // Only master process prints error
            cerr << "Usage: mpirun -np <num_processes> ./traffic_mpi <input_file> <top_N>\n";
        }
        MPI_Finalize();
        return 1;
    }

    string filename = argv[1];  // Input filename
    int top_n = stoi(argv[2]);  // Number of top congested lights to display

    if (rank == 0) {
        // ================= MASTER PROCESS =================
        ifstream infile(filename);
        if (!infile.is_open()) {
            cerr << "Failed to open file " << filename << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Read all lines from input file
        vector<string> lines;
        string line;
        while (getline(infile, line)) {
            lines.push_back(line);
        }

        // Calculate work distribution among workers
        int num_workers = size - 1;  // All processes except master (rank 0)
        int chunk_size = lines.size() / num_workers;
        int extra = lines.size() % num_workers;
        int index = 0;

        // Distribute work to workers
        for (int i = 1; i <= num_workers; ++i) {
            // Determine chunk size for this worker
            int send_count = chunk_size + (i <= extra ? 1 : 0);
            
            // First send the count of lines to process
            MPI_Send(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            
            // Then send each line of data
            for (int j = 0; j < send_count; ++j) {
                string& entry = lines[index++];
                MPI_Send(entry.c_str(), entry.size() + 1, MPI_CHAR, i, 0, MPI_COMM_WORLD);
            }
        }

        // ========== COLLECT AND PROCESS RESULTS ==========
        map<string, map<string, int>> global_map;  // hour -> light_id -> count

        // Receive results from each worker
        for (int i = 1; i <= num_workers; ++i) {
            int entry_count;
            MPI_Recv(&entry_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Receive each entry from worker
            for (int j = 0; j < entry_count; ++j) {
                vector<char> hour_buf(100), light_buf(100);
                int count;

                MPI_Recv(hour_buf.data(), 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(light_buf.data(), 100, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Aggregate counts in global map
                global_map[hour_buf.data()][light_buf.data()] += count;
            }
        }

        // Create human-readable names for traffic lights
        unordered_map<string, string> readable_names;
        int light_number = 1;
        for (const auto& [hour, light_data] : global_map) {
            for (const auto& [light_id, _] : light_data) {
                if (readable_names.find(light_id) == readable_names.end()) {
                    readable_names[light_id] = "Traffic Light " + to_string(light_number++);
                }
            }
        }

        // ========== OUTPUT RESULTS ==========
        // Print all traffic records
        cout << "\nAll Traffic Records:\n";
        for (const auto& [hour, light_data] : global_map) {
            for (const auto& [light_id, count] : light_data) {
                cout << "[" << hour << "] " 
                     << readable_names[light_id] << " - " 
                     << count << " cars\n";
            }
        }

        // Print top N congested lights per hour
        cout << "\nTop " << top_n << " Congested Traffic Lights Per Hour:\n";
        for (const auto& [hour, light_data] : global_map) {
            cout << "\n[Hour: " << hour << "] Top " << top_n << " traffic lights:\n";
            
            // Convert map to vector and sort by count (descending)
            vector<pair<string, int>> light_counts(light_data.begin(), light_data.end());
            sort(light_counts.begin(), light_counts.end(), [](const auto& a, const auto& b) {
                return b.second < a.second;
            });

            // Print top N entries
            for (int i = 0; i < min(top_n, static_cast<int>(light_counts.size())); ++i) {
                const auto& [light_id, count] = light_counts[i];
                cout << "  " << readable_names[light_id] 
                     << " - " << count << " cars\n";
            }
        }

    } else {
        // ================= WORKER PROCESS =================
        int recv_count;
        // Receive number of lines to process from master
        MPI_Recv(&recv_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Local data structure to store counts
        map<string, map<string, int>> local_map;  // hour -> light_id -> count

        // Process each received line
        for (int i = 0; i < recv_count; ++i) {
            vector<char> buf(256);
            MPI_Recv(buf.data(), 256, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Parse the line: timestamp light_id count
            istringstream ss(string(buf.data()));
            string timestamp, light_id;
            int count;
            
            if (ss >> timestamp >> light_id >> count) {
                if (!timestamp.empty() && !light_id.empty()) {
                    local_map[timestamp][light_id] += count;
                }
            }
        }

        // Count total entries to send back
        int entry_count = 0;
        for (const auto& [hour, light_data] : local_map) {
            entry_count += light_data.size();
        }

        // First send the count of entries
        MPI_Send(&entry_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

        // Then send each entry (hour, light_id, count)
        for (const auto& [hour, light_data] : local_map) {
            for (const auto& [light_id, count] : light_data) {
                MPI_Send(hour.c_str(), hour.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                MPI_Send(light_id.c_str(), light_id.size() + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
                MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    // Clean up MPI environment
    MPI_Finalize();
    return 0;
}
