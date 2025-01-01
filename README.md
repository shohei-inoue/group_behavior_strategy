# group behavior strategy

## プロジェクト構成

- `src/main.py`: アプリケーションのエントリーポイント。
- `Dockerfile`: Dockerイメージを構築するための設定ファイル。Python 3.12.1をベースに、NumPy、Gym、Matplotlibをインストールします。
- `docker-compose.yml`: Docker Composeの設定ファイル。サービスの定義やボリュームの設定を行い、コンテナを簡単に立ち上げることができます。

## セットアップ手順

1. このリポジトリをクローンします。
   ```
   git clone <repository-url>
   ```

2. DockerとDocker Composeがインストールされていることを確認します。

3. Docker Composeを使用してコンテナをビルドし、起動します。
   ```
   docker-compose up --build
   ```

4. アプリケーションが実行。

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。