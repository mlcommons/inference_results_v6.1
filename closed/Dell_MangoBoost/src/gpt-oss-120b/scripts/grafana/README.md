### Start Grafana and SQLite services

In the `docker-compose.yaml` update the path of the volumes which will be mounted inside the container if needed: e.g. the path of the database file.

```
services:
  grafana:
    ...
    volumes:
      - /home/amd/inference/grafana_db:/var/lib/grafana/datasources
    ...
  sqlite_web:
    ...
    volumes:
      - /home/amd/inference/grafana_db:/data
```

Run `docker compose up -d` to create and start the sqlite and grafana services in the background.

Visit `localhost:4141` for Grafana, the default user/pw is `admin/admin`.

Maybe need to forward the ports 4141 (Grafana), 7171 (SQLite viewer) to access them.

### Add data source

- Navigate to Connections -> Data sources
- Press button **Add data source**, select SQLite, set the **Path** to `/var/lib/grafana/datasources/database.db`, then press button **Save & test**.

### Add dashboards

To add some pre-created dashboard
- Navigate to Dashboards
- Press the button **New**, select **Import**.
- Import the json files from the `dashboards` folder.
- Select the datasource below, then press **Import**.

### Github Action runners / machines

The Github Action runners are responsible for running your jobs in Github Actions workflows.

To run jobs on a specific machine, ensure that machine is registered as runner: https://github.com/AMD-MLPerf/mlperf-inference/settings/actions/runners

Press **New self-hosted runner** and follow the instructions to register a new one.
