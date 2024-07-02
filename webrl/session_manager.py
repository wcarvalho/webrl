class SessionManager:

    def __init__(self, debug: bool = False):
        self.idx = 0
        self.state = None
        self.debug = debug
        self.index_called = False
        self.experiment_called = False

    def initialize(
        self,
        stage_manager: StageManager,
        load_user: bool = True,
        initial_template: str = 'consent.html',
        experiment_fn_name: str = 'experiment',
    ):
        session.permanent = load_user
        self.index_called = True
        if load_user:
            if 'unique_id' not in session:
                print("Loading new user")
                SessionManager.reset_user()
            else:
                print(f"loading prior user: {session['unique_id']}")
                print(f"Redirecting to: {experiment_fn_name}")
                return stage_manager.redirect(experiment_fn_name)
        else:
            print("Resetting user")
            session.clear()
            SessionManager.reset_user()

        return stage_manager.render(template=initial_template)


    def split_rng(self):
        rng = jax_utils.unserialize_rng(session['rng'])
        rng, rng_ = jax.random.split(rng)
        session['rng'] = jax.serialize_rng(rng)
        self.update_stage(rng=rng)
        return rng_

    def save_state(key, default=None):
        value = session.get(key, default)
        session[key] = value
        return value

    def upload_data(self):
        import ipdb; ipdb.set_trace()

    @staticmethod
    def reset_user():
        unique_id = random.getrandbits(32)
        user_seed = int(unique_id)
        session['unique_id'] = unique_id
        session['user_seed'] = user_seed

    @staticmethod
    def __setitem__(key, value):
        session[key] = value

    @staticmethod
    def set(key, value):
        session[key] = value

    @staticmethod
    def __getitem__(key):
        value = session[key]
        return value

    @staticmethod
    def get(key, default=None, overwrite: bool = True):
        value = session.get(key, default)
        if overwrite:
            session[key] = value
        return value

    @staticmethod
    def clear():
        session.clear()


