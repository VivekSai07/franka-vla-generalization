import copy
import unittest


def manifest():
    return dict(fps=30, camera_size=[512, 512], camera_names=['front_cam', 'gripper_cam'],
                splits={'train': list(range(300)), 'validation': list(range(300, 420))},
                episodes=[dict(episode_index=i, split='train' if i < 300 else 'validation',
                               pick='red', place='green', colors=['red', 'green'], frames=528,
                               instruction='pick up the red cube and place it on top of the green cube')
                          for i in range(420)])


class SelectionTests(unittest.TestCase):
    def test_only_approved_episodes_selected(self):
        from experiments.xvla_trial.contract import select_entries
        self.assertEqual([e['episode_index'] for e in select_entries(manifest())],
                         list(range(9)) + list(range(300, 304)))

    def test_wrong_task_split_or_cube_count_rejected(self):
        from experiments.xvla_trial.contract import select_entries
        for key, value in [('pick', 'green'), ('colors', ['red', 'green', 'blue']),
                           ('split', 'train'), ('frames', 527), ('instruction', 'other')]:
            m = manifest()
            m['episodes'][300][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                select_entries(m)

    def test_duplicate_and_split_leakage_rejected(self):
        from experiments.xvla_trial.contract import select_entries
        for duplicate in [True, False]:
            m = manifest()
            if duplicate:
                m['episodes'][1] = copy.deepcopy(m['episodes'][0])
            else:
                m['splits']['train'].append(300)
            with self.assertRaises(ValueError):
                select_entries(m)

    def test_complete_chunks_only_no_validation_rows(self):
        from experiments.xvla_trial.contract import indices
        episodes = [dict(episode_index=i, dataset_from_index=i*528, dataset_to_index=(i+1)*528)
                    for i in range(420)]
        rows, starts = indices(episodes)
        self.assertEqual(rows, list(range(4752)))
        self.assertEqual(len(starts), 4491)
        self.assertEqual(starts[:499], list(range(499)))
        self.assertEqual(starts[-1], 4722)
        self.assertFalse(set(rows) & set(range(158400, 160512)))

    def test_inconsistent_episode_offsets_rejected(self):
        from experiments.xvla_trial.contract import indices
        episodes = [dict(episode_index=i, dataset_from_index=i*528, dataset_to_index=(i+1)*528)
                    for i in range(420)]
        episodes[300]['dataset_from_index'] = 0
        episodes[300]['dataset_to_index'] = 528
        with self.assertRaises(ValueError):
            indices(episodes)