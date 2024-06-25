
const TopicHeader = ({ title, description }) => {
    return (
        <header className="bg-nav-color/[.70] px-10 py-10">
            <h1 className="text-5xl mb-3 text-blue-400">
                {title}
            </h1>
            <p>
                {description}
            </p>
        </header>
    )
}

export default TopicHeader;