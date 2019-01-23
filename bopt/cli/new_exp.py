import yaml
import os
import inspect

def run(args):
    script_dir = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))

    yaml_template_fname = os.path.join(
        script_dir,
        "..",
        "templates",
        "new_experiment.yml"
    )

    with open(yaml_template_fname, "r") as f:
        yaml_template = f.read()

    yaml_template = yaml_template.replace("EXPERIMENT_NAME", args.NAME)

    print(yaml_template)

    if not os.path.exists(args.DIR):
        print(f"Directory {args.DIR} doesn't exist, creating.")
        os.mkdir(args.DIR)

    target_fname = os.path.join(args.DIR, f"{args.NAME}.yml")

    with open(target_fname, "wt") as f:
        f.write(yaml_template)

    os.mkdir(os.path.join(args.DIR, args.NAME))
