# Contributing to the Open Mining Format

There are many ways to contribute to OMF:
- [**Become a member** of GMG](#membership)
- [**Raise issues** for any questions/feedback/problems you have](#issues)
- [**Write code** to directly move the format forward](#code)
- [**Provide documentation** and examples](#docs)

## <a name="membership"></a>High-level involvement through Global Mining Guidelines Group

The Open Mining Format has been created through the Data Exchange for
Mine Software Project under the
[Data Access and Usage Working Group](https://gmggroup.org/groups/data-access-and-usage-dau/).
This sub-committee is involved with all aspects of the success of OMF,
including industry engagement and organization, technical design and
development, and outreach and marketing.
[Learn more about GMG](https://gmggroup.org/) or
[become a member](https://gmggroup.org/about-us/membership/)

## <a name="issues"></a>Raising questions, feedback, problems

Given the *open* nature of OMF, all technical conversations and
development happen, here, on GitHub, visible to everyone. To
participate, all you need is a [free GitHub account](https://github.com/join);
there is no requirement to be a member of GMG or even part of
the mining community.

Any questions/feedback/problems should be raised as
[issues](https://github.com/gmggroup/omf/issues). You may comment on
existing, relevant issues or
[create a new issue](https://github.com/gmggroup/omf/issues/new). There
is no restriction on what issues are "supposed to" look like; this is
a place for anyone to voice anything about OMF. Examples include:
- detailed technical questions around implementation
- problems encountered when attempting to support OMF
- confusion around the documentation
- feature requests for the format
- questions around the high-level goal of an open standard
- suggestions around non-technical aspects, such as marketing and engagement
- etc!

If the question is non-technical, follow up may happen outside of
GitHub, but this is a place to start.

To ensure the OMF community remains welcoming and productive, please read and
follow our [Code of Conduct](code_of_conduct.md).

## <a name="code"></a>Contributing to code development

Anyone can submit pull requests to the OMF repository. Preferably these
are related to an
[existing bug](https://github.com/gmggroup/omf/issues?q=is%3Aopen+is%3Aissue+label%3Abug)
or a feature included in an
[upcoming milestone](https://github.com/gmggroup/omf/milestones).
[Low-hanging-fruit issues](https://github.com/gmggroup/omf/issues?q=is%3Aopen+is%3Aissue+label%3A%22%3Aarrow_down%3A+%3Agreen_apple%3A%22)
are great for first-time contributors. If the solution to the issue is
unclear, please follow up and ask for clarification; if nothing else,
it's useful to know what people are working on. If your pull request
does not have an existing issue, consider [creating an issue](#issues)
first, just to add context and promote discussion.

When working on your contribution, you may
[fork](https://help.github.com/en/articles/fork-a-repo) the OMF repo to
your personal or company GitHub organization and develop there.
Alternatively, if you are interested in being identified as a contributor
to the [GMG GitHub organization](https://github.com/gmggroup),
reach out to [Heather Turnbull](mailto:hturnbull@gmggroup.org),
the Operations Manager at GMG (note: this is distinct from
[GMG membership](#membership)). Once you are a contributor on GitHub, you may
[create feature branches](https://help.github.com/en/articles/creating-and-deleting-branches-within-your-repository)
directly in the GMG OMF repository.

When creating a branch, consider naming it in the format
`GH-##/human_readable_description`, where "##" is the related
issue number. Strive for as much inter-linking as possible of pull
requests, issues numbers, commits, etc.

When submitting a pull request, please base off the `dev` branch.
Contributions will be collected here, then version-bumped and deployed
via pull request from `dev` to `master` as appropriate.

Finally, everyone appreciates unit tests, code documentation, and
consistent style (just run [black](https://black.readthedocs.io/en/stable/)).
And, to ensure the OMF community remains welcoming
and productive, read and follow our [Code of Conduct](code_of_conduct.md).

### <a name="docs"></a>Contributing documentation and examples

The most useful contributions for the success of OMF are documentation
and examples that can be shared with everyone. To use the format,
people must understand the format. Documentation comes in many forms;
it can include:
- Technical documentation of the code and API
- Description of how the API relates to real objects
- Workflows describing specific implementations of OMF in prose, code,
  or screenshots
- Example OMF files, ideally along side source files of other formats
  and description of the import/export process.
- etc!

Contributing documentation and examples is also more flexible than
code contributions. Documentation hosted on
[readthedocs](https://omf.readthedocs.io/en/stable/) is directly built
from the GitHub repository, so you may contribute documentation there
by submitting a PR. However, it is also entirely valid to
[create a new issue](https://github.com/gmggroup/omf/issues/new) and
attach any files or text you have there. Somebody will take those
submissions and put them in the appropriate place.

By documenting and highlighting diverse, successful OMF use-cases, we
are able to demonstrate early industry engagement, and this will promote
further adoption.

