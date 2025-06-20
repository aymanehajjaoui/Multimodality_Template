/*TCP.cpp*/

#include "TCP.hpp"

int initialize_tcp(const std::string &ip, int port)
{
    int sock;
    struct sockaddr_in server_addr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Socket creation failed");
        return -1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip.c_str(), &server_addr.sin_addr) <= 0)
    {
        perror("Invalid IP address");
        close(sock);
        return -1;
    }

    while (!stop_program.load() &&
           connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0)
    {
        perror("Connection failed, retrying...");
        sleep(1);
    }

    if (stop_program.load())
    {
        close(sock);
        std::cerr << "[TCP] Connection aborted due to stop request\n";
        return -1;
    }

    return sock;
}

int get_gateway_ip(char *ip_buffer, size_t size)
{
    FILE *fp = popen("ip route | grep default | awk '{print $3}'", "r");
    if (!fp)
        return -1;
    if (fgets(ip_buffer, size, fp) == NULL)
    {
        pclose(fp);
        return -1;
    }
    ip_buffer[strcspn(ip_buffer, "\n")] = 0;
    pclose(fp);
    return 0;
}
