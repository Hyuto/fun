import { rm } from "fs";
import yargs from "yargs";
import { hideBin } from "yargs/helpers";
import { execa } from "execa";
import Listr from "listr";

const argv = yargs(hideBin(process.argv))
  .usage("$0 <command>")
  .command("setup", "Setup python and dependencies")
  .parse();

if (argv.setup) {
  console.log("Setting up python and dependencies");
  new Listr([
    {
      title: "Making python virtual environments",
      task: () => {
        let command = "";

        switch (process.platform) {
          case "win32":
            command = "py -m venv webview-env";
            break;
          case "linux":
            command = "python3 -m venv webview-env";
            break;
          default:
            command = "python3 -m venv webview-env";
            break;
        }

        return execa(command);
      },
    },
    {
      title: "Installing python dependencies",
      task: () => {
        let command = "";

        switch (process.platform) {
          case "win32":
            command = ".\\webview-env\\Scripts\\python -m pip install -r requirements.txt";
            break;
          case "linux":
            command = "./webview-env/bin/python3 -m pip install -r requirements.txt";
            break;
          default:
            command = "./webview-env/bin/python -m pip install -r requirements.txt";
            break;
        }

        return execa(command);
      },
    },
  ])
    .run()
    .catch((err) => {
      console.error(err);
    });
}

if (argv.build) {
  console.log("Building Application");
  new Listr([
    {
      title: "Cleaning main directory",
      task: () =>
        Promise.all([
          rm(".parcel-cache", { recursive: true, force: true }, (err) => {
            if (err) {
              throw new Error(err);
            }
          }),
          rm("dist", { recursive: true, force: true }, (err) => {
            if (err) {
              throw new Error(err);
            }
          }),
        ]),
    },
    {
      title: "Building React Application",
      task: () => execa("parcel build --public-url ."),
    },
    {
      title: "Make Executable",
      task: () => {
        let command = "";

        switch (process.platform) {
          case "win32":
            command = ".\\webview-env\\Scripts\\pyinstaller main.spec";
            break;
          case "linux":
            command = "./webview-env/bin/pyinstaller main.spec";
            break;
          default:
            throw new Error("Not Supported OS!");
            break;
        }

        return execa(command);
      },
    },
  ])
    .run()
    .catch((err) => {
      console.error(err);
    });
}
