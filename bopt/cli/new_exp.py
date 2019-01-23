import yaml
import os
import inspect

def run(args) -> None:
    script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    yaml_template_fname = os.path.join(
        script_dir,
        "..",
        "templates",
        "new_experiment.yml"
    )

    with open(yaml_template_fname, "r") as f:
        yaml_template = f.read()

    name = os.path.basename(args.DIR)

    yaml_template = yaml_template.replace("EXPERIMENT_NAME", name)

    if not os.path.exists(args.DIR):
        print(f"Directory {args.DIR} doesn't exist, creating.")
        os.mkdir(args.DIR)

    target_fname = os.path.join(args.DIR, "config.yml")

    with open(target_fname, "wt") as f:
        f.write(yaml_template)

    print(f"Created a new experiment at {args.DIR}")
