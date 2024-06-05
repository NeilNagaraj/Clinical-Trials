# !/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import webbrowser
from pathlib import Path
from time import sleep
from uuid import uuid4

import click
import requests
import trio
from alive_progress import alive_bar
from uvicorn import Config

from databutton.decorators.jobs.schedule import Scheduler
from databutton.server.dev import DatabuttonRunner
from databutton.utils import (
    create_databutton_cloud_project,
    create_databutton_config,
    get_auth_token,
    get_build_logs,
    get_databutton_config,
    get_databutton_login_info,
    get_databutton_login_path,
)
from databutton.utils.deploy import (
    create_archive,
    get_build_id_from_deployment,
    listen_to_build,
    upload_archive,
)
from databutton.utils.uvicorn_in_mem import get_threaded_uvicorn

from .__init__ import __version__

logger = logging.getLogger("databutton.cli")

LOGGING_LEVELS = {
    0: logging.NOTSET,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}


@click.group()
@click.option("--verbose", "-v", count=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, verbose: int):
    ctx.ensure_object(dict)
    """Run databutton."""
    # Use the verbosity count to determine the logging level...
    if verbose > 0:
        logger.setLevel(
            LOGGING_LEVELS[verbose] if verbose in LOGGING_LEVELS else logging.DEBUG
        )
    ctx.obj["VERBOSE"] = verbose


@cli.command()
def version():
    """Get the library version."""
    click.echo(click.style(f"{__version__}", bold=True))


@cli.command()
@click.argument(
    "project-directory",
    required=True,
)
@click.option(
    "-t",
    "--template",
    required=False,
    default="hello-databutton",
    help="A template to bootstrap off of.",
)
@click.option(
    "-n",
    "--name",
    required=False,
    help="The name of your Databutton project (on databutton.com)",
)
@click.option("--non-interactive", default=False, is_flag=True)
@click.pass_context
def create(
    ctx: click.Context,
    project_directory: Path,
    template: str,
    name: str,
    non_interactive: bool,
):
    """
    Create a Databutton project in the provided project-directory

    PROJECT_DIRECTORY is the name of the directory for your Databutton project.

    Examples:

    \b
    # Create a new Databutton project in the `my-new-project` directory
    databutton create my-new-project

    \b
    # Create a new Databutton project with a custom name
    databutton create my-new-project -n "My new Databutton project"

    \b
    # Create a new Databutton project in the current directory
    databutton create .

    \b
    # Create a new Databutton project from a template
    databutton create my-new-project --template=sales-forecasting

    """
    project_directory = Path(project_directory).resolve()
    project_name = name if name else str(project_directory.as_posix()).split("/")[-1]

    if project_directory.exists():
        if len(list(project_directory.iterdir())) > 0:
            click.secho("❌ The target directory isn't empty.")
            exit(1)
    project_directory_relative = project_directory.relative_to(Path.cwd())
    login_info = get_databutton_login_info()
    if not login_info:
        click.echo()
        click.secho("It doesn't seem like you have a user with Databutton.")
        click.secho(
            f"In order to use {click.style('databutton deploy', fg='cyan')} and "
            + f"{click.style('db.storage.dataframes', fg='cyan')}, "
            "you need a user and a project in Databutton."
        )
        should_login = (
            False
            if non_interactive
            else click.confirm("\nWould you like to create a user now?", default=True)
        )
        click.echo()
        if should_login:
            ctx.invoke(login)
            login_info = get_databutton_login_info()

    res = requests.get(
        f"https://storage.googleapis.com/databutton-app-templates/{template}.tar.gz",
        stream=True,
    )
    if not res.ok:
        click.secho(
            f"❌ Failed to get template {template}, are you sure you typed an existing template?"
        )
        exit(1)
    if res.ok:
        with tempfile.TemporaryFile() as tmpfile:
            tmpfile.write(res.raw.read())
            tmpfile.seek(0)
            with tarfile.open(fileobj=tmpfile) as t:
                t.extractall(project_directory)

        if login_info:
            should_create_project = (
                False
                if non_interactive
                else click.confirm(
                    "\nDo you want to also create a project on databutton.com?\nYou can always do databutton init later.",
                    default=True,
                )
            )
            if should_create_project:
                project_id = create_databutton_cloud_project(project_name)
                create_databutton_config(
                    name=project_name,
                    uid=project_id,
                    project_directory=project_directory,
                )
            else:
                create_databutton_config(
                    name=project_name,
                    uid=str(uuid4()),
                    project_directory=project_directory,
                )
        else:
            create_databutton_config(
                name=project_name, uid=str(uuid4()), project_directory=project_directory
            )
        click.echo()
        click.echo(
            f"Success! Created project {click.style(project_name, fg='cyan')} in {project_directory_relative}"
        )
        click.echo("Inside that directory, you can run several commands:")
        click.echo()

        commands = [
            ("databutton start", "Starts the development server."),
            (
                "databutton deploy",
                "Deploy your project to databutton.com.\n    "
                + "Note that in order to deploy you need a user and a project in Databutton.",
            ),
            (
                "databutton build",
                "Bundles the project and generates the necessary files for production.",
            ),
        ]

        for cmd, desc in commands:
            click.echo(f"  {click.style(cmd, fg='cyan')}")
            click.echo(f"    {click.style(desc)}")
            click.echo()

        click.echo("We suggest you begin by typing:\n")

        click.echo(f"  {click.style('cd', fg='cyan')} {project_directory_relative}")
        click.secho("  pip install -r requirements.txt", fg="cyan")
        click.secho("  databutton start", fg="cyan")
        click.secho("  databutton deploy", fg="cyan")

        click.echo()
        click.secho(
            f"You can always see the available commands by typing {click.style('databutton --help', fg='cyan')}."
            + f"\nDocumentation is available by typing {click.style('databutton docs', fg='cyan')}."
        )

        click.echo("\nHappy building!\n")


@cli.command()
@click.option("--debug", default=False, type=bool, show_default=True, is_flag=True)
def start(debug=False):
    """Run the Databutton development server"""
    runner = DatabuttonRunner()

    try:
        trio.run(
            runner.run,
            debug,
            # instruments=[Tracer(logger_name="databutton.start")],
        )
    except KeyboardInterrupt:
        logger.debug("Closing scheduler")
        sys.exit(0)


@cli.command()
def login():
    """Login to Databutton"""
    click.echo(click.style("Opening browser to authenticate.."))

    dir_path = Path(__file__).parent
    app_dir = dir_path / "auth"
    sys.path.insert(0, str(app_dir.resolve()))
    login_url = "http://localhost:8008"
    server = get_threaded_uvicorn(
        Config("server:app", port=8008, log_level="error", reload=False)
    )

    # Remove login details
    shutil.rmtree(get_databutton_login_path(), ignore_errors=True)
    with server.run_in_thread():
        webbrowser.open(login_url)
        login_path = get_databutton_login_path()
        while not (os.path.exists(login_path) and len(os.listdir(login_path)) > 0):
            sleep(1)
        # Wait a little bit so the user gets a response
    click.secho("Logged in!")


@cli.command()
@click.pass_context
def deploy(ctx: click.Context):
    """Deploy your project to Databutton"""

    login_info = get_databutton_login_info()
    if login_info is None:
        click.secho("❌ You're not authenticated to Databutton, yet.")
        click.secho(
            f"Type {click.style('databutton login', fg='cyan')} to login or create a user."
        )
        exit(1)

    try:
        config = get_databutton_config(retries=2)
    except FileNotFoundError:
        click.secho(
            "❌ Can't find a Databutton config file. Are you in the correct folder?"
        )
        click.secho(
            f"Type {click.style('databutton init', fg='cyan')} to "
            + "create a new Databutton project in this directory."
        )
        exit(1)

    deployment_id = str(uuid4())

    click.echo()
    click.echo(click.style(f"=== Deploying to {config.name}", fg="green"))
    ctx.invoke(build)
    click.echo(click.style("i packaging components...", fg="cyan"))

    with tempfile.NamedTemporaryFile() as tmpfile:
        create_archive(tmpfile, source_dir=Path.cwd(), config=config)
        click.echo(click.style("i done packaging components", fg="green"))
        click.echo(click.style("i uploading components", fg="cyan"))
        try:
            upload_archive(config, deployment_id, tmpfile)
        except Exception:
            click.secho("❌ Could not upload, talk to someone at Databutton to debug...")
            exit(1)

        click.echo(click.style("i finished uploading components", fg="green"))
        click.echo(click.style("i cleaning up", fg="cyan"))

    click.echo(
        click.style(
            "i waiting for deployment to be ready, this can take a few minutes...",
            fg="cyan",
        )
    )
    click.secho("i deploying...", fg="cyan")
    click.echo()
    click.secho(
        "i you can close this window if you want, the deploy will continue in the cloud.",
        fg="cyan",
    )

    build_id = get_build_id_from_deployment(deployment_id)
    build_logs = get_build_logs(build_id)
    click.echo()
    click.echo(f"Build logs are available at {build_logs}")
    click.echo()

    with alive_bar(
        stats=False, monitor=False, monitor_end=False, elapsed=False, title=""
    ) as bar:
        for status in listen_to_build(deployment_id):
            bar.text = f"status: {status.capitalize() if status else 'Queued'}"
            bar()

    if status == "SUCCESS":
        click.echo(click.style("✅ Done!", fg="green"))
        click.echo()
        styled_url = click.style(
            f"https://next.databutton.com/projects/{config.uid}", fg="cyan"
        )
        click.secho(f"You can now go to \n\t{styled_url}")
        click.echo()
    elif status == "FAILURE":
        click.echo(click.style("❌ Error deploying...", fg="red"))
        click.echo()
    elif status == "CANCELLED":
        click.secho(
            "Your build was cancelled. A databutler probably did that on purpose."
        )
        click.secho("You should reach out!")


@cli.command()
def build():
    """Build the project, built components will be found in .databutton"""
    click.echo(click.style("i building project", fg="cyan"))
    from databutton.utils.build import generate_components

    click.echo(click.style("i generating components", fg="cyan"))
    artifacts = generate_components()
    click.echo(click.style("i finished building project in .databutton", fg="green"))
    return artifacts


@cli.command()
def serve():
    """Starts a web server for production."""
    click.echo(click.style("=== Serving"))
    import uvicorn

    port = os.environ["PORT"] if "PORT" in os.environ else 8000
    uvicorn.run(
        "databutton.server.prod:app",
        reload=False,
        port=int(port),
        host="0.0.0.0",
    )


@cli.command("init")
@click.option("--name", help="Name of the project")
def init(name: str):
    """Creates a new project in Databutton and writes to databutton.json"""
    login_info = get_databutton_login_info()
    if login_info is None:
        click.secho(
            f"You're not logged in. Log in with {click.style('databutton login', fg='cyan')} first."
        )
        exit(1)
    if os.path.exists("databutton.json"):
        click.secho("There is already a databutton.json file in this directory")
        project_config = get_databutton_config()
        click.secho(f"  id: {click.style(project_config.uid, fg='cyan')}")
        click.secho(f"  name: {click.style(project_config.name, fg='cyan')}")
        should_overwrite = click.confirm(
            "Do you want to create a new project and overwrite it?", default=True
        )
        if not should_overwrite:
            click.secho("You did not create a new Databutton project.")
            exit(0)
    token = get_auth_token()
    if not name:
        name = click.prompt("Choose a name for your databutton project", type=str)
    res = requests.post(
        "https://europe-west1-databutton.cloudfunctions.net/createOrUpdateProject",
        json={"name": name},
        headers={"Authorization": f"Bearer {token}"},
    )

    res_json = res.json()
    new_id = res_json["id"]
    config = create_databutton_config(name, new_id)
    click.secho(
        f"✅ Created project {name}",
        fg="green",
    )
    click.echo()
    styled_url = click.style(
        f"https://next.databutton.com/projects/{new_id}", fg="cyan"
    )
    click.echo(f"You can check out your project on \n\n  {styled_url}\n")
    click.secho(
        f"Type {click.style('databutton deploy', fg='cyan')} to deploy your project."
    )
    return config


@cli.command()
def docs():
    """Launches https://docs.databutton.com"""
    click.launch("https://docs.databutton.com")


@cli.command()
def logout():
    """Removes all Databutton login info"""
    login_info = get_databutton_login_info()
    if login_info is not None:
        shutil.rmtree(get_databutton_login_path(), ignore_errors=True)
        click.secho("Logged out")
        click.secho(
            f"You can always log in again with {click.style('databutton login', fg='cyan')}"
        )
    else:
        click.secho(
            f"No Databutton user found, did you mean {click.style('databutton login', fg='cyan')}?"
        )


@cli.command()
def whoami():
    """Shows the logged in user"""
    import jwt

    try:
        token = get_auth_token()
    except Exception:
        click.secho("No user found.")
        click.secho(f"Log in first with {click.style('databutton login', fg='cyan')}.")
        exit(1)
    decoded = jwt.decode(token, options={"verify_signature": False})
    click.secho("Found logged in user")
    keys_to_print = ["name", "email", "user_id"]
    for k in keys_to_print:
        click.secho(f"  {k}: {click.style(decoded[k], fg='cyan')}")
    click.echo()


@cli.command()
def schedule():
    """Starts and runs the scheduler"""
    click.secho("=== Starting the databutton scheduler", fg="cyan")

    Scheduler.create()
