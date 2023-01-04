# Market Making via Reinforcement Learning

Training reinforcement learning agents to provide liquidity in high-frequency order-driven markets as in the paper Market Making via Reinforcement Learning by Spooner et al.
It is set up to use data provided by [LOBSTER](https://lobsterdata.com/). 


### Setting up the database with a docker container
The data from Lobster's files is stored in a postgres database within docker.
Therefore it is necessary to set up a postgres database locally using the following steps:
1. [Download and install docker](https://docs.docker.com/engine/install/ubuntu/)
2. Create a docker container running postgres by running in a command prompt:

    ```docker run --name limitorderbook -e POSTGRES_USER=root -e POSTGRES_PASSWORD=root -e POSTGRES_DB=limitorderbook -p 5432:5432 -v /data:/var/lib/limitorderbook/data -d postgres -c shared_buffers=1GB```
3. Check it is running in Docker Desktop Containers