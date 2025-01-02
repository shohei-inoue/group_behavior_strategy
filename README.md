# group behavior strategy

## プロジェクト構成

- `src/main.py`: アプリケーションのエントリーポイント。
- `envs/group_behavior_strategy_env.py`: 強化学習環境を定義するファイル。
- `agents/red_agents.py`: 強化学習エージェントを定義するファイル。
- `robots/red.py`: ロボットの動作を定義するファイル。
- `Dockerfile`: Dockerイメージを構築するための設定ファイル。Python 3.12.1をベースに、NumPy、Gym、Matplotlibをインストールします。
- `docker-compose.yml`: Docker Composeの設定ファイル。サービスの定義やボリュームの設定を行い、コンテナを簡単に立ち上げることができます。
- `.env`: データベース接続情報などの環境変数を定義するファイル。
- `requirements.txt`: プロジェクトで使用するPythonパッケージのリスト。

## セットアップ手順

1. このリポジトリをクローンします。
   ```sh
   git clone <repository-url>
2. DockerとDocker Composeがインストールされていることを確認します。
3. .envファイルをプロジェクトのルートディレクトリに作成し, データベース接続情報を記述します。
```
DATABASE_URL=postgresql://user:password@localhost:5432/mydatabase
```
4. Docker Composeを使用してコンテナをビルドし、起動します。
```
docker-compose up --build
```