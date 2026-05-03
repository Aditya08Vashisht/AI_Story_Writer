from story_mvp.generator import generate_story_piece


def test_hook_generation_returns_demo_fields():
    result = generate_story_piece(
        {
            "mode": "hook",
            "genre": "thriller",
            "tone": "suspenseful",
            "language": "hinglish",
            "idea": "A writer receives a call from a listener who has been missing for years.",
            "characters": "Rhea, Kabir",
        }
    )

    assert result["mode"] == "hook"
    assert result["genre"] == "thriller"
    assert "output" in result
    assert "Rhea" in result["output"]
    assert result["story_bible"]["primary_characters"] == ["Rhea", "Kabir"]
    assert result["safety"]["status"] == "clear"


def test_invalid_options_fall_back_to_safe_defaults():
    result = generate_story_piece(
        {
            "mode": "unknown",
            "genre": "unknown",
            "tone": "unknown",
            "idea": "",
        }
    )

    assert result["mode"] == "continue"
    assert result["genre"] == "thriller"
    assert result["tone"] == "cinematic"
    assert result["output"]


def test_scene_expansion_longer_than_hook():
    hook = generate_story_piece(
        {
            "mode": "hook",
            "genre": "family drama",
            "tone": "emotional",
            "idea": "A daughter finds her late mother's diary.",
            "characters": "Anaya, Dev",
        }
    )
    scene = generate_story_piece(
        {
            "mode": "expand",
            "genre": "family drama",
            "tone": "emotional",
            "idea": "A daughter finds her late mother's diary.",
            "characters": "Anaya, Dev",
            "length": "long",
        }
    )

    assert len(scene["output"]) > len(hook["output"])
    assert "Anaya" in scene["output"]


def test_story_generation_across_modes_and_genres():
    cases = [
        {
            "mode": "hook",
            "genre": "thriller",
            "tone": "suspenseful",
            "language": "hinglish",
            "idea": "A writer receives a call from a listener who has been missing for years.",
            "characters": "Rhea, Kabir",
        },
        {
            "mode": "hook",
            "genre": "horror",
            "tone": "mythic",
            "language": "english",
            "idea": "A prayer bell rings inside an abandoned recording studio.",
            "characters": "Riya, Kabir",
        },
        {
            "mode": "hook",
            "genre": "romance",
            "tone": "cinematic",
            "language": "english",
            "idea": "Two former radio hosts meet during a blackout and one carries an unsent confession.",
            "characters": "Aarav, Meera",
        },
        {
            "mode": "expand",
            "genre": "fantasy",
            "tone": "cinematic",
            "language": "english",
            "length": "long",
            "idea": "A map redraws itself when a storyteller remembers the dragon's name.",
            "characters": "Mira, Arjun",
        },
        {
            "mode": "expand",
            "genre": "family drama",
            "tone": "emotional",
            "language": "hindi",
            "length": "medium",
            "idea": "A daughter returns home for her brother's wedding and finds a diary hidden in the prayer room.",
            "characters": "Anaya, Dev",
        },
        {
            "mode": "expand",
            "genre": "horror",
            "tone": "mythic",
            "language": "english",
            "length": "short",
            "idea": "A mirror covered in cloth begins to reflect a scene from tomorrow.",
            "characters": "Riya, Kabir",
        },
        {
            "mode": "continue",
            "genre": "comedy",
            "tone": "witty",
            "language": "english",
            "length": "medium",
            "idea": "Two rival radio hosts scramble to save a live prank that accidentally exposes a secret.",
            "characters": "Veena, Sameer",
        },
        {
            "mode": "continue",
            "genre": "thriller",
            "tone": "suspenseful",
            "language": "english",
            "length": "long",
            "idea": "The safe witness finally decides to speak, and every truth changes the broadcast.",
            "characters": "Rhea, Kabir",
        },
        {
            "mode": "continue",
            "genre": "fantasy",
            "tone": "cinematic",
            "language": "english",
            "length": "medium",
            "idea": "A storyteller discovers the kingdom's fate is tied to the song in her voice note.",
            "characters": "Mira, Arjun",
        },
        {
            "mode": "continue",
            "genre": "romance",
            "tone": "emotional",
            "language": "english",
            "length": "medium",
            "idea": "A city blackout turns a late-night recording session into a choice that will decide both their stories.",
            "characters": "Amaya, Jai",
        },
    ]

    for case in cases:
        result = generate_story_piece(case)

        assert result["mode"] == case["mode"]
        assert result["genre"] == case["genre"]
        assert result["output"]
        assert result["pitch"]
        assert result["story_bible"]["primary_characters"] == [name.strip() for name in case["characters"].split(",")]
        assert result["safety"]["status"] == "clear"
